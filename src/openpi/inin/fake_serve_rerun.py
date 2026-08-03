"""Publish the model's supervision targets of one dataset episode to rerun.

The policy is trained on *relative* action chunks (``DeltaActions`` subtracts
the observation pose from the first six action dimensions), while the robot is
sent *absolute* chunks. ``openpi.inin.serve`` closes that gap at inference time
with the pose of the observation it just ran on; here the very same
reanchoring is done with the observation pose read straight from the dataset,
so what shows up in rerun is exactly what training saw and what the robot
would have received had the model reproduced its targets perfectly.

The supervision chunk is not recomputed by hand: the training config's
transform chain (repack -> IninInputs -> DeltaActions) is applied as-is, with
only normalization and the model transforms skipped, so a drift between this
view and training is impossible by construction.

rerun runs as a gRPC server and never spawns a viewer; connect with
``rerun rerun+http://<host>:<port>/proxy``.

Example:
    uv run python -m openpi.inin.fake_serve_rerun --episode 0 --stride 10 \
        --data-root /mnt/cpfs/zbl-cpfs-new/dataset/harryjhou --grpc-port 9876
"""

import dataclasses
import logging
import pathlib
import time

import numpy as np
from scipy.spatial.transform import Rotation
import tyro

from openpi import transforms as _transforms
from openpi.inin import conversions
from openpi.training import config as _config

try:
    import rerun as rr
    import rerun.blueprint as rrb
except ImportError as exc:  # pragma: no cover - environment problem, not a code path
    raise ImportError(
        "rerun-sdk is required by openpi.inin.fake_serve_rerun; it normally comes in with lerobot. "
        "Install it with `uv pip install 'rerun-sdk>=0.23'`."
    ) from exc

logger = logging.getLogger(__name__)

_WORLD_PATH_COLOR = (255, 170, 0)
_OBS_LINK_COLOR = (255, 80, 80)
_RELATIVE_PATH_COLOR = (80, 200, 255)


@dataclasses.dataclass(frozen=True)
class Args:
    # Training config providing the dataset id and the transform chain.
    config: str = "pi05_inin_ur5"
    # Overrides the config's repo_id; the transforms stay the same.
    repo_id: str | None = None
    # Dataset lives in <data_root>/<repo_id>. Falls back to HF_LEROBOT_HOME.
    data_root: pathlib.Path | None = None

    # Episode to replay.
    episode: int = 0
    # First / last (exclusive) frame of the episode; None means the whole episode.
    start_frame: int = 0
    end_frame: int | None = None
    # Publish every Nth frame; the chunks of skipped frames are never read.
    stride: int = 10
    # Hard cap on the number of published frames, applied after the stride.
    max_frames: int | None = None

    # Chunk length. Defaults to the config's action_horizon (50 for pi05_inin_ur5).
    action_horizon: int | None = None
    # Publish every Nth step within a chunk; 1 keeps all of them.
    chunk_stride: int = 1

    # rerun gRPC server; the viewer connects with rerun+http://<host>:<port>/proxy.
    grpc_port: int = 9876
    app_id: str = "openpi_inin_supervision"
    server_memory_limit: str = "25%"
    # Throttle publishing to this rate so an attached viewer sees frames roll in.
    # None dumps the whole episode at once.
    stream_hz: float | None = None

    # Axis lengths separating the semantics: origin > obs pose > chunk step.
    origin_axis_length: float = 0.30
    obs_axis_length: float = 0.12
    chunk_axis_length: float = 0.03


@dataclasses.dataclass(frozen=True)
class _Frame:
    """One published frame: the observation and both views of its chunk."""

    frame_index: int
    timestamp_s: float
    # Observation pose in the dataset's on-disk representation.
    obs_xyz: np.ndarray
    obs_quat_xyzw: np.ndarray
    obs_gripper: float
    # Supervision target: (N, 7) xyz + rotvec + gripper, relative to the obs pose.
    relative: np.ndarray
    # Same chunk rendered as poses: (N, 8) xyz + quat + gripper about the origin.
    relative_wire: np.ndarray
    # Reanchored with the obs pose: (N, 8) xyz + quat + gripper, the robot's format.
    reanchored_wire: np.ndarray
    # The dataset's own absolute chunk, (N, 8), for the roundtrip check.
    raw_wire: np.ndarray
    # Chunk steps that get a coordinate frame in rerun; the scalars stay on the
    # full chunk so --chunk-stride only thins the 3D view.
    step_indices: np.ndarray
    # Chunk steps that LeRobot clamped to the episode's last frame.
    pad_steps: int


def _load_dataset(args: Args, data_config, action_horizon: int):
    """Open the episode with the same chunking the training data loader uses."""
    from lerobot.common.datasets import lerobot_dataset

    repo_id = args.repo_id or data_config.repo_id
    if repo_id is None:
        raise ValueError(f"config {args.config!r} has no repo_id; pass --repo-id")
    root = str(pathlib.Path(args.data_root) / repo_id) if args.data_root is not None else None

    meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id, root=root)
    if not 0 <= args.episode < meta.total_episodes:
        raise ValueError(f"episode {args.episode} out of range; {repo_id} has {meta.total_episodes}")

    # ``episodes=[k]`` keeps only that episode in memory and makes the dataset
    # index the episode-local frame index. Chunks running past the end are
    # clamped by LeRobot and flagged in ``action_is_pad``, exactly as in training.
    dataset = lerobot_dataset.LeRobotDataset(
        repo_id,
        root=root,
        episodes=[args.episode],
        delta_timestamps={
            key: [t / meta.fps for t in range(action_horizon)] for key in data_config.action_sequence_keys
        },
    )
    logger.info(
        "loaded %s episode %d: %d frames fps=%.0f horizon=%d",
        repo_id,
        args.episode,
        len(dataset),
        meta.fps,
        action_horizon,
    )
    return dataset


def _build_frame(item: dict, supervision, reanchor, chunk_stride: int) -> _Frame:
    """Run the training transforms on one dataset item and reanchor the result."""
    state14 = np.asarray(item["observation.state"], dtype=np.float64)
    raw_wire = np.asarray(item["action"], dtype=np.float64)

    model_inputs = supervision(item)
    state7 = np.asarray(model_inputs["state"], dtype=np.float32)
    relative = np.asarray(model_inputs["actions"], dtype=np.float32)

    # AbsoluteActions adds the state back in place, so the supervision chunk
    # must not be the array handed to it.
    reanchored = reanchor({"state": state7, "actions": relative.copy()})["actions"]
    reanchored_wire = conversions.rotvec_chunk_to_quat_chunk(reanchored)

    return _Frame(
        frame_index=int(item["frame_index"]),
        timestamp_s=float(item["timestamp"]),
        obs_xyz=state14[6:9],
        obs_quat_xyzw=state14[9:13],
        obs_gripper=float(state14[13]),
        relative=relative,
        # The relative rotation is an element-wise rotvec difference, not a
        # composed rotation; treating it as one is what makes it drawable.
        relative_wire=conversions.rotvec_chunk_to_quat_chunk(relative),
        reanchored_wire=reanchored_wire,
        raw_wire=raw_wire,
        step_indices=np.arange(0, len(relative), chunk_stride),
        pad_steps=int(np.asarray(item["action_is_pad"]).sum()),
    )


def _log_pose(entity: str, xyz: np.ndarray, quat_xyzw: np.ndarray, axis_length: float) -> None:
    rr.log(
        entity,
        rr.Transform3D(
            translation=np.asarray(xyz, dtype=np.float32),
            # A bare xyzw array rather than rr.Quaternion: the wrapper's arrow
            # conversion calls np.asarray(copy=...), which only exists in numpy 2,
            # and rerun swallows the failure as a warning with the rotation dropped.
            quaternion=np.asarray(quat_xyzw, dtype=np.float32),
            axis_length=axis_length,
        ),
    )


def _log_chunk(
    root: str,
    chunk_wire: np.ndarray,
    steps: np.ndarray,
    axis_length: float,
    color: tuple[int, int, int],
) -> None:
    """Log the selected chunk steps as coordinate frames, plus the polyline joining them."""
    for step in steps:
        row = chunk_wire[step]
        _log_pose(f"{root}/{step:03d}", row[:3], row[3:7], axis_length)
    rr.log(
        f"{root}/path",
        rr.LineStrips3D([np.asarray(chunk_wire[steps, :3], dtype=np.float32)], colors=[color], radii=[0.0015]),
    )


def _log_scalars(frame: _Frame) -> None:
    """Numbers that make a broken chunk obvious without reading the 3D view."""
    positions = frame.reanchored_wire[:, :3]
    rotations = Rotation.from_quat(frame.reanchored_wire[:, 3:7])

    steps_xyz = np.linalg.norm(np.diff(positions, axis=0), axis=1) if len(positions) > 1 else np.zeros(1)
    steps_rot = np.degrees((rotations[1:] * rotations[:-1].inv()).magnitude()) if len(positions) > 1 else np.zeros(1)

    # Reanchoring inverts the delta exactly, so anything above float32 noise
    # here means the representation conversions are lossy for this frame.
    raw_rotations = Rotation.from_quat(frame.raw_wire[:, 3:7])
    roundtrip_rot = np.degrees((rotations * raw_rotations.inv()).magnitude())

    rr.log("scalars/gripper/obs", rr.Scalars(frame.obs_gripper))
    rr.log("scalars/gripper/action_first", rr.Scalars(float(frame.reanchored_wire[0, 7])))

    rr.log("scalars/chunk_step/translation_max", rr.Scalars(float(steps_xyz.max())))
    rr.log("scalars/chunk_step/translation_mean", rr.Scalars(float(steps_xyz.mean())))
    rr.log("scalars/chunk_step/rotation_deg_max", rr.Scalars(float(steps_rot.max())))

    rr.log("scalars/offset/obs_to_action", rr.Scalars(float(np.linalg.norm(positions[0] - frame.obs_xyz))))
    rr.log("scalars/offset/relative_last", rr.Scalars(float(np.linalg.norm(frame.relative[-1, :3]))))

    rr.log(
        "scalars/roundtrip/translation_max",
        rr.Scalars(float(np.abs(positions - frame.raw_wire[:, :3]).max())),
    )
    rr.log("scalars/roundtrip/rotation_deg_max", rr.Scalars(float(roundtrip_rot.max())))
    rr.log(
        "scalars/roundtrip/gripper_max",
        rr.Scalars(float(np.abs(frame.reanchored_wire[:, 7] - frame.raw_wire[:, 7]).max())),
    )

    rr.log("scalars/pad_steps", rr.Scalars(float(frame.pad_steps)))


def _log_frame(frame: _Frame, args: Args) -> None:
    rr.set_time("frame", sequence=frame.frame_index)
    rr.set_time("episode_time", duration=frame.timestamp_s)

    _log_pose("world/obs", frame.obs_xyz, frame.obs_quat_xyzw, args.obs_axis_length)
    _log_chunk("world/action", frame.reanchored_wire, frame.step_indices, args.chunk_axis_length, _WORLD_PATH_COLOR)
    rr.log(
        "world/obs_to_action",
        rr.LineStrips3D(
            [np.stack([frame.obs_xyz, frame.reanchored_wire[0, :3]]).astype(np.float32)],
            colors=[_OBS_LINK_COLOR],
            radii=[0.002],
        ),
    )
    _log_chunk("relative/action", frame.relative_wire, frame.step_indices, args.chunk_axis_length, _RELATIVE_PATH_COLOR)

    _log_scalars(frame)


def _log_static(args: Args) -> None:
    # Both roots carry their own view coordinates: a 3D view resolves them from
    # its origin entity, and the two views are rooted at "world" / "relative".
    for root in ("world", "relative"):
        rr.log(root, rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        rr.log(
            f"{root}/origin",
            rr.Transform3D(translation=[0.0, 0.0, 0.0], axis_length=args.origin_axis_length),
            static=True,
        )


def _blueprint() -> rrb.Blueprint:
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(origin="/world", name="absolute (obs + reanchored chunk)"),
            rrb.Spatial3DView(origin="/relative", name="relative chunk (supervision)"),
            rrb.Vertical(
                rrb.TimeSeriesView(origin="/scalars/chunk_step", name="chunk step size"),
                rrb.TimeSeriesView(origin="/scalars/roundtrip", name="reanchor roundtrip error"),
                rrb.TimeSeriesView(origin="/scalars/gripper", name="gripper"),
                rrb.TimeSeriesView(origin="/scalars/offset", name="offsets"),
            ),
            column_shares=[3, 2, 2],
        ),
        collapse_panels=True,
    )


def _frame_indices(args: Args, num_frames: int) -> range | list[int]:
    if args.stride < 1:
        raise ValueError("--stride must be >= 1")
    if args.chunk_stride < 1:
        raise ValueError("--chunk-stride must be >= 1")
    end = num_frames if args.end_frame is None else min(args.end_frame, num_frames)
    indices = range(max(args.start_frame, 0), end, args.stride)
    if args.max_frames is not None:
        return list(indices)[: args.max_frames]
    return indices


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", force=True)

    train_config = _config.get_config(args.config)
    # Norm stats are not needed: normalization and the model transforms are
    # skipped so the chunks stay in metres and radians.
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    action_horizon = args.action_horizon or train_config.model.action_horizon

    dataset = _load_dataset(args, data_config, action_horizon)
    supervision = _transforms.compose(
        [
            _transforms.PromptFromLeRobotTask(dataset.meta.tasks),
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
        ]
    )
    reanchor = _transforms.compose(data_config.data_transforms.outputs)

    rr.init(args.app_id, spawn=False)
    # Serve before logging so late-connecting viewers get the whole episode
    # from the server's buffer.
    uri = rr.serve_grpc(
        grpc_port=args.grpc_port,
        default_blueprint=_blueprint(),
        server_memory_limit=args.server_memory_limit,
    )
    logger.info("rerun serving at %s; connect with: rerun %s", uri, uri)
    _log_static(args)

    indices = _frame_indices(args, len(dataset))
    period_s = 1.0 / args.stream_hz if args.stream_hz else 0.0
    published = 0
    for index in indices:
        frame = _build_frame(dataset[index], supervision, reanchor, args.chunk_stride)
        _log_frame(frame, args)
        published += 1
        if published == 1 or published % 20 == 0:
            logger.info(
                "published frame_index=%d chunk=%s pad_steps=%d roundtrip_xyz=%.2e m",
                frame.frame_index,
                frame.reanchored_wire.shape,
                frame.pad_steps,
                float(np.abs(frame.reanchored_wire[:, :3] - frame.raw_wire[:, :3]).max()),
            )
        if period_s:
            time.sleep(period_s)

    logger.info("published %d frame(s) of episode %d; serving until Ctrl-C", published, args.episode)
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        logger.info("stopping rerun server")


if __name__ == "__main__":
    main(tyro.cli(Args))

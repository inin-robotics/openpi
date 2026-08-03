"""Fake inin workstation speaking the bc-ur5-v2 schema, for smoke tests.

Unlike inin-stream's generic fake workstation, this one honours the full BC
collection contract: unit-quaternion TCP poses, a non-empty ``task`` in
episode_start metadata, a ``frames`` count in episode_end metadata, and
288x384 camera images. In ``infer`` mode it reports received action chunks
instead of waiting for collection commits.

``infer_sync`` exercises the synchronous contract of scripts/inin_serve_sync.py:
one observation, one chunk, and the next observation only after the chunk has
been executed. It deliberately uses nothing beyond inin-stream's released
workstation API, so it doubles as the reference implementation for the real
robot client (see examples/inin/sync_client_task.md).

``replay_sync`` is the same round trip for scripts/inin_replay_sync.py, with a
ReplayStamp in place of the observation. It replays until the server answers
``replay_complete`` rather than for a fixed number of rounds.

Examples:
    uv run scripts/inin_fake_workstation.py --server 127.0.0.1:43601 \
        --mode collect --episodes 2 --frames 40
    uv run scripts/inin_fake_workstation.py --server 127.0.0.1:43602 --mode infer
    uv run scripts/inin_fake_workstation.py --server 127.0.0.1:43603 \
        --mode infer_sync --episodes 1 --frames 20
    uv run scripts/inin_fake_workstation.py --server 127.0.0.1:43605 \
        --mode replay_sync --episodes 1
"""

import dataclasses
import logging
import pathlib
import tempfile
import time
from typing import Literal

from inin_stream.schema import builtin_schema_path
from inin_stream.testing import write_workstation_config
from inin_stream.workstation import TimedAction
from inin_stream.workstation import WorkstationStreamClient
import numpy as np
from scipy.spatial.transform import Rotation
import tyro

# latest_only would overwrite a pending observation, which breaks the
# synchronous contract: a dropped request is never answered.
_QUEUE_POLICIES = {
    "collect": "lossless_blocking",
    "infer": "latest_only",
    "infer_sync": "lossless_blocking",
    "replay_sync": "lossless_blocking",
}
_RUN_MODES = {
    "collect": "bc_collect",
    "infer": "bc_infer",
    "infer_sync": "bc_infer_sync",
    "replay_sync": "bc_replay_sync",
}


@dataclasses.dataclass(frozen=True)
class Args:
    server: str = "127.0.0.1:43601"
    mode: Literal["collect", "infer", "infer_sync", "replay_sync"] = "collect"
    episodes: int = 2
    # Observations per episode; in infer_sync mode this is the number of rounds.
    frames: int = 40
    fps: float = 10.0
    task: str = "pick up the corn"
    commit_timeout_s: float = 120.0
    # infer_sync / replay_sync: how long to wait for the chunk answering one request.
    action_timeout_s: float = 30.0
    # infer_sync / replay_sync: rate at which a chunk's steps are executed,
    # matching the schema's action_inference.target_period_hz. Only used to fake
    # the execution time of a received chunk.
    exec_hz: float = 30.0
    # replay_sync: give up if the server never says replay_complete. The server
    # decides how long an episode is, so the client needs its own upper bound.
    max_rounds: int = 10_000


def _tcp_pose(t: float) -> np.ndarray:
    """A smooth unit-quaternion TCP trajectory (crosses the pi boundary)."""
    xyz = np.array([0.4 + 0.05 * np.sin(t), 0.1 * np.cos(t), 0.3 + 0.02 * t])
    quat = Rotation.from_rotvec(np.array([0.0, 0.0, 1.0]) * (2.9 + 0.5 * np.sin(0.5 * t))).as_quat()
    return np.concatenate([xyz, quat]).astype(np.float32)


def _observation(stream: WorkstationStreamClient, rng: np.random.Generator, t: float, stamp_ns: int):
    pose = _tcp_pose(t)
    gripper = np.array([0.5 + 0.4 * np.sin(t)], dtype=np.float32)
    image = rng.integers(0, 256, size=(288, 384, 3), dtype=np.uint8)
    return stream.prepare_observation(
        images={"camera.wrist.rgb": image, "camera.base.rgb": image},
        tensors={
            "robot.tcp_pose": pose,
            "robot.gripper_openness": gripper,
            "robot.joint_pos": rng.normal(scale=0.1, size=(6,)).astype(np.float32),
        },
        actions={
            # Commanded pose: slightly ahead of the measured pose.
            "tcp_pose_cmd": _tcp_pose(t + 0.1),
            "gripper_openness_cmd": gripper,
        },
        stamp_ns=stamp_ns,
        anchor_observation_key="camera.wrist.rgb",
    )


def _wait_for_chunk(stream: WorkstationStreamClient, obs_seq: int, timeout_s: float) -> TimedAction | None:
    """Block until the chunk answering ``obs_seq`` arrives, or time out.

    ``ActionMessage.source_seq_used`` carries the obs_seq the server ran on, so
    it is what pairs a chunk with its request. ``get_actions_until()`` returns
    and clears everything buffered (it ignores the timestamp it is given), so
    each call must scan the whole batch; anything that does not match is a
    leftover from an earlier round and is discarded.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        for timed in stream.get_actions_until(time.monotonic_ns()):
            if timed.source_seq_used == obs_seq:
                return timed
            logging.warning(
                "discarding stale chunk seq=%d for obs_seq=%d while waiting for %d",
                timed.action_seq,
                timed.source_seq_used,
                obs_seq,
            )
        time.sleep(0.002)
    return None


def _run_episode_sync(stream: WorkstationStreamClient, args: Args, rng: np.random.Generator, index: int) -> None:
    """One observation, one chunk, execute it fully, repeat."""
    event = stream.start_episode(metadata={"task": args.task, "source": "inin_fake_workstation"})
    logging.info("sync episode %d started: %s", index, event.episode_id)
    start = time.monotonic()
    for round_index in range(args.frames):
        # A fresh round must not see a chunk left over from a timed-out one.
        stream.clear_action_buffer()
        frame = _observation(stream, rng, t=time.monotonic() - start, stamp_ns=time.time_ns())
        if not stream.publish_observation(frame):
            raise RuntimeError(f"observation publish failed at round {round_index}")

        wait_start = time.monotonic()
        timed = _wait_for_chunk(stream, frame.obs_seq, timeout_s=args.action_timeout_s)
        if timed is None:
            raise RuntimeError(f"no chunk for obs_seq={frame.obs_seq} within {args.action_timeout_s}s")
        wait_ms = (time.monotonic() - wait_start) * 1e3
        chunk = timed.action
        if chunk.ndim != 2 or chunk.shape[1] != 8:
            raise RuntimeError(f"unexpected chunk shape: {chunk.shape}")

        # A real client would stream these poses to the controller and wait for
        # the motion to finish; here the execution time is just simulated.
        exec_s = chunk.shape[0] / args.exec_hz
        time.sleep(exec_s)
        logging.info(
            "round %d: obs_seq=%d chunk=%s wait=%.0f ms exec=%.0f ms",
            round_index,
            frame.obs_seq,
            chunk.shape,
            wait_ms,
            exec_s * 1e3,
        )

    stream.end_episode(reason="success", metadata={"frames": str(args.frames)})
    logging.info("sync episode %d finished: %d round(s)", index, args.frames)


def _wait_for_chunk_or_completion(
    stream: WorkstationStreamClient, stamp_seq: int, timeout_s: float
) -> TimedAction | None:
    """Block until the stamp is answered by a chunk or by ``replay_complete``.

    Returns the chunk, or None once the replay is complete. The pairing checks
    mirror the real client's: a chunk from a future stamp means the server
    answered a stamp twice or ran ahead, which cannot be recovered from.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        for control in stream.get_controls():
            if control.type == "replay_complete":
                return None
            logging.warning("ignoring control %r while waiting for stamp_seq=%d", control.type, stamp_seq)
        for timed in stream.get_actions_until(time.monotonic_ns()):
            if timed.source_seq_used > stamp_seq:
                raise RuntimeError(
                    f"action_protocol_desync: chunk for stamp_seq={timed.source_seq_used} while waiting for {stamp_seq}"
                )
            if timed.source_seq_used < stamp_seq:
                logging.warning("discarding stale chunk for stamp_seq=%d", timed.source_seq_used)
                continue
            return timed
        time.sleep(0.002)
    raise RuntimeError(f"action_timeout: no answer for stamp_seq={stamp_seq} within {timeout_s}s")


def _run_episode_replay_sync(stream: WorkstationStreamClient, args: Args, rng: np.random.Generator, index: int) -> None:
    """One stamp, one chunk, execute it fully, repeat until replay_complete."""
    del rng
    event = stream.start_episode(metadata={"task": args.task, "source": "inin_fake_workstation"})
    logging.info("replay episode %d started: %s", index, event.episode_id)
    steps_executed = 0
    for stamp_seq in range(args.max_rounds):
        # Anything still buffered means the previous stamp was answered more
        # than once, which the real client treats as duplicate_action.
        leftover = stream.get_actions_until(time.monotonic_ns())
        if leftover:
            raise RuntimeError(
                f"duplicate_action: {len(leftover)} extra chunk(s) buffered before stamp_seq={stamp_seq}"
            )
        # A fresh round must not see a chunk left over from a timed-out one.
        stream.clear_action_buffer()
        if not stream.publish_replay_stamp(stamp_ns=time.time_ns(), stamp_seq=stamp_seq):
            raise RuntimeError(f"replay stamp publish failed at stamp_seq={stamp_seq}")

        wait_start = time.monotonic()
        timed = _wait_for_chunk_or_completion(stream, stamp_seq, timeout_s=args.action_timeout_s)
        if timed is None:
            stream.end_episode(reason="success", metadata={"frames": str(steps_executed)})
            logging.info(
                "replay episode %d finished: rounds=%d steps=%d reason=replay_complete",
                index,
                stamp_seq,
                steps_executed,
            )
            return
        wait_ms = (time.monotonic() - wait_start) * 1e3
        chunk = timed.action
        if chunk.ndim != 2 or chunk.shape[1] != 8:
            raise RuntimeError(f"unexpected chunk shape: {chunk.shape}")

        # A real client would stream these poses to the controller and wait for
        # the motion to finish; here the execution time is just simulated.
        exec_s = chunk.shape[0] / args.exec_hz
        time.sleep(exec_s)
        steps_executed += chunk.shape[0]
        logging.info(
            "executing complete replay chunk action_seq=%d stamp_seq=%d steps=%d wait=%.0f ms",
            timed.action_seq,
            stamp_seq,
            chunk.shape[0],
            wait_ms,
        )

    raise RuntimeError(f"server never sent replay_complete within {args.max_rounds} rounds")


def _run_episode(stream: WorkstationStreamClient, args: Args, rng: np.random.Generator, index: int) -> None:
    event = stream.start_episode(metadata={"task": args.task, "source": "inin_fake_workstation"})
    logging.info("episode %d started: %s", index, event.episode_id)
    period_s = 1.0 / args.fps
    chunks_received = 0
    base_stamp = time.time_ns()
    for seq in range(args.frames):
        frame = _observation(stream, rng, t=seq * period_s, stamp_ns=base_stamp + int(seq * 1e9 / args.fps))
        if not stream.publish_observation(frame):
            logging.warning("publish failed at seq=%d", seq)
        if args.mode == "infer":
            actions = stream.get_actions_until(time.monotonic_ns())
            for timed in actions:
                chunks_received += 1
                logging.info(
                    "received action chunk seq=%d shape=%s mode=%s",
                    timed.action_seq,
                    timed.action.shape,
                    timed.metadata.get("action_mode", ""),
                )
        time.sleep(period_s)

    if args.mode == "infer":
        # Give the server a moment to flush the last chunk.
        time.sleep(1.0)
        for timed in stream.get_actions_until(time.monotonic_ns() + 10_000_000_000):
            chunks_received += 1
            logging.info(
                "received action chunk seq=%d shape=%s mode=%s",
                timed.action_seq,
                timed.action.shape,
                timed.metadata.get("action_mode", ""),
            )
        stream.end_episode(reason="success", metadata={"frames": str(args.frames)})
        logging.info("episode %d finished: chunks_received=%d", index, chunks_received)
        if chunks_received == 0:
            raise RuntimeError("no action chunks received in infer mode")
        return

    stream.end_episode(reason="success", metadata={"frames": str(args.frames)})
    control = stream.wait_for_control(
        "collection_episode_committed",
        episode_id=event.episode_id,
        timeout_s=args.commit_timeout_s,
    )
    if control is None:
        raise RuntimeError(f"episode {event.episode_id} was not committed within {args.commit_timeout_s}s")
    logging.info("episode %d committed: params=%s", index, dict(control.params))


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    rng = np.random.default_rng(42)
    schema_path = builtin_schema_path("bc-ur5-v2")
    with tempfile.TemporaryDirectory(prefix="inin_fake_ws_") as tmp:
        config_path = write_workstation_config(
            pathlib.Path(tmp) / "workstation.yaml",
            args.server,
            policy=_QUEUE_POLICIES[args.mode],
            run_mode=_RUN_MODES[args.mode],
        )
        stream = WorkstationStreamClient(config_path=config_path, schema_path=schema_path)
        stream.start()
        run_episode = {
            "infer_sync": _run_episode_sync,
            "replay_sync": _run_episode_replay_sync,
        }.get(args.mode, _run_episode)
        try:
            for index in range(args.episodes):
                run_episode(stream, args, rng, index)
        finally:
            stream.stop()
    logging.info("fake workstation done: %d episode(s) in %s mode", args.episodes, args.mode)


if __name__ == "__main__":
    main(tyro.cli(Args))

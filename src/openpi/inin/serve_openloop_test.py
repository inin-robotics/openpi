# ruff: noqa: SLF001

from types import SimpleNamespace

from inin_stream.common.serialization import ObservationFrame
import numpy as np

from openpi.inin.serve_openloop import LeRobotEpisodeSource
from openpi.inin.serve_openloop import OpenLoopInferenceCallbacks
from openpi.inin.serve_openloop import pose_error


def _state(x: float = 0.0) -> np.ndarray:
    return np.array([0, 0, 0, 0, 0, 0, x, 0, 0, 0, 0, 0, 1, 0.5], dtype=np.float32)


def _dataset(length: int = 21) -> list[dict]:
    return [
        {
            "frame_index": index,
            "state": _state(index * 0.001),
            "image": np.zeros((8, 8, 3), dtype=np.uint8),
            "wrist_image": np.zeros((8, 8, 3), dtype=np.uint8),
            "prompt": "stack the blocks",
            "actions": np.zeros((50, 8), dtype=np.float32),
        }
        for index in range(length)
    ]


def _transform(item: dict) -> dict:
    return dict(item)


def _stream_frame(*, obs_seq: int, x: float = 0.0) -> ObservationFrame:
    return ObservationFrame(
        robot_id="robot",
        run_id="run",
        episode_id="episode",
        obs_seq=obs_seq,
        stamp_ns=obs_seq,
        anchor_observation_key="camera.wrist.rgb",
        tensors={"robot.tcp_pose": np.array([x, 0, 0, 0, 0, 0, 1], dtype=np.float32)},
        actions={},
        scalars={},
        images={},
        image_metadata={},
    )


class _Policy:
    def __init__(self):
        self.observations = []

    def infer(self, obs: dict) -> dict:
        self.observations.append(obs)
        return {"actions": np.zeros((50, 7), dtype=np.float32)}


class _Server:
    def __init__(self):
        self.actions = []
        self.controls = []

    def send_action(self, **kwargs) -> None:
        self.actions.append(kwargs)

    def send_control(self, **kwargs) -> None:
        self.controls.append(kwargs)


def _callbacks(*, execute: bool, end_frame: int | None = None):
    source = LeRobotEpisodeSource(
        _dataset(),
        _transform,
        start_frame=0,
        end_frame=end_frame,
        frame_step=10,
    )
    policy = _Policy()
    server = _Server()
    callbacks = OpenLoopInferenceCallbacks(
        policy,
        source,
        exec_steps=10,
        execute=execute,
        max_translation_m=0.03,
        max_rotation_deg=15.0,
    )
    callbacks.attach_stream_server(server)
    callbacks.on_schema_accepted(SimpleNamespace(robot_id="robot", run_id="run", run_mode="infer"))
    callbacks.on_episode_start(SimpleNamespace(episode_id="episode", metadata={}, reason=""))
    return callbacks, source, policy, server


def test_source_uses_dataset_prompt_and_drops_supervision():
    source = LeRobotEpisodeSource(_dataset(), _transform, frame_step=10)
    frame = source.current()

    assert frame.prompt == "stack the blocks"
    assert "actions" not in frame.observation
    assert frame.dataset_index == 0


def test_execute_advances_by_exec_steps_and_resets_per_episode():
    callbacks, source, policy, server = _callbacks(execute=True)

    callbacks._infer_and_send(_stream_frame(obs_seq=1, x=0.0))
    callbacks._infer_and_send(_stream_frame(obs_seq=2, x=0.01))

    assert [call["source_seq_used"] for call in server.actions] == [1, 2]
    assert source.index == 20
    assert [round(float(obs["state"][6]) * 1000) for obs in policy.observations] == [0, 10]

    callbacks.on_episode_start(SimpleNamespace(episode_id="episode", metadata={}, reason=""))
    assert source.index == 0


def test_dry_run_infers_without_sending_action():
    callbacks, source, policy, server = _callbacks(execute=False)

    callbacks._infer_and_send(_stream_frame(obs_seq=1))

    assert len(policy.observations) == 1
    assert server.actions == []
    assert source.index == 10


def test_pose_mismatch_sends_stop_and_does_not_advance():
    callbacks, source, _, server = _callbacks(execute=True)

    callbacks._infer_and_send(_stream_frame(obs_seq=1, x=0.5))

    assert server.actions == []
    assert len(server.controls) == 1
    assert server.controls[0]["control_type"] == "stop_remote_action"
    assert source.index == 0


def test_dataset_exhaustion_sends_stop_once():
    callbacks, _, _, server = _callbacks(execute=True, end_frame=1)

    callbacks._infer_and_send(_stream_frame(obs_seq=1))
    callbacks._infer_and_send(_stream_frame(obs_seq=2))
    callbacks._infer_and_send(_stream_frame(obs_seq=3))

    assert len(server.actions) == 1
    assert len(server.controls) == 1
    assert "exhausted" in server.controls[0]["message"]


def test_pose_error_handles_quaternion_double_cover():
    state = _state()
    live = np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)

    translation_m, rotation_deg = pose_error(live, state)

    assert translation_m == 0.0
    assert rotation_deg == 0.0

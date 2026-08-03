from types import SimpleNamespace

import numpy as np
import pytest

from openpi.inin.replay_sync import ReplaySyncCallbacks


def _actions(length: int = 25) -> np.ndarray:
    """A (length, 8) trajectory whose first column identifies the frame."""
    actions = np.zeros((length, 8), dtype=np.float32)
    actions[:, 0] = np.arange(length, dtype=np.float32)
    actions[:, 6] = 1.0  # unit quaternion (qw)
    return actions


class _Server:
    def __init__(self):
        self.actions = []
        self.controls = []

    def send_action(self, **kwargs) -> None:
        self.actions.append(kwargs)

    def send_control(self, **kwargs) -> None:
        self.controls.append(kwargs)

    @property
    def message_count(self) -> int:
        return len(self.actions) + len(self.controls)


def _stamp(stamp_seq: int, *, episode_id: str = "episode") -> SimpleNamespace:
    return SimpleNamespace(episode_id=episode_id, stamp_seq=stamp_seq, stamp_ns=1000 + stamp_seq)


def _callbacks(*, exec_steps: int = 10, start_frame: int = 0, end_frame: int | None = None, length: int = 25):
    server = _Server()
    callbacks = ReplaySyncCallbacks(
        _actions(length),
        exec_steps=exec_steps,
        start_frame=start_frame,
        end_frame=end_frame,
    )
    callbacks.attach_stream_server(server)
    callbacks.on_schema_accepted(SimpleNamespace(robot_id="robot", run_id="run", run_mode="bc_replay_sync"))
    callbacks.on_episode_start(SimpleNamespace(episode_id="episode", metadata={}, reason=""))
    return callbacks, server


def test_each_stamp_gets_exactly_one_paired_chunk():
    callbacks, server = _callbacks()

    for stamp_seq in range(3):
        callbacks.on_replay_stamp(_stamp(stamp_seq))

    assert server.message_count == 3
    assert [call["source_seq_used"] for call in server.actions] == [0, 1, 2]
    assert [call["obs_stamp_ns"] for call in server.actions] == [1000, 1001, 1002]
    assert {call["action_mode"] for call in server.actions} == {"absolute_tcp_gripper_chunk"}
    assert {call["robot_id"] for call in server.actions} == {"robot"}


def test_chunks_are_contiguous_and_never_overlap():
    callbacks, server = _callbacks()

    for stamp_seq in range(2):
        callbacks.on_replay_stamp(_stamp(stamp_seq))

    replayed = [call["action"][:, 0].tolist() for call in server.actions]
    assert replayed[0] == list(range(10))
    assert replayed[1] == list(range(10, 20))


def test_tail_chunk_is_short_rather_than_padded():
    callbacks, server = _callbacks()

    for stamp_seq in range(3):
        callbacks.on_replay_stamp(_stamp(stamp_seq))

    assert server.actions[-1]["action"].shape == (5, 8)
    assert server.actions[-1]["action"][:, 0].tolist() == [20, 21, 22, 23, 24]


def test_replay_complete_is_sent_only_after_the_data_runs_out():
    callbacks, server = _callbacks()

    for stamp_seq in range(3):
        callbacks.on_replay_stamp(_stamp(stamp_seq))
    assert server.controls == []

    callbacks.on_replay_stamp(_stamp(3))

    assert len(server.actions) == 3
    assert len(server.controls) == 1
    assert server.controls[0]["control_type"] == "replay_complete"
    assert server.controls[0]["params"] == {"episode_id": "episode"}


def test_stamps_after_completion_keep_answering_one_for_one():
    callbacks, server = _callbacks(length=10)

    for stamp_seq in range(3):
        callbacks.on_replay_stamp(_stamp(stamp_seq))

    assert server.message_count == 3
    assert len(server.controls) == 2


def test_frame_range_restricts_the_replay():
    callbacks, server = _callbacks(start_frame=5, end_frame=17)

    assert callbacks.total_rounds == 2
    for stamp_seq in range(3):
        callbacks.on_replay_stamp(_stamp(stamp_seq))

    assert [call["action"][:, 0].tolist() for call in server.actions] == [
        list(range(5, 15)),
        [15, 16],
    ]
    assert len(server.controls) == 1


def test_cursor_restarts_with_stamp_seq_on_a_second_episode():
    callbacks, server = _callbacks()

    callbacks.on_replay_stamp(_stamp(0))
    callbacks.on_episode_end(SimpleNamespace(episode_id="episode", reason="success"))
    callbacks.on_episode_start(SimpleNamespace(episode_id="episode-2", metadata={}, reason=""))
    callbacks.on_replay_stamp(_stamp(0, episode_id="episode-2"))

    assert server.actions[0]["action"][:, 0].tolist() == server.actions[1]["action"][:, 0].tolist()
    assert callbacks.sent_chunks == 2
    assert callbacks._episode_chunks == 1  # noqa: SLF001


def test_failure_sends_nothing_so_the_client_can_time_out():
    callbacks, server = _callbacks()

    callbacks.on_replay_stamp(_stamp(-1))

    assert server.message_count == 0


def test_zero_quaternion_is_rejected_before_serving():
    actions = _actions()
    actions[7, 3:7] = 0.0

    with pytest.raises(ValueError, match="zero TCP quaternion"):
        ReplaySyncCallbacks(actions)


def test_empty_frame_range_is_rejected():
    with pytest.raises(ValueError, match="empty"):
        ReplaySyncCallbacks(_actions(), start_frame=10, end_frame=10)

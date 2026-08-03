"""Synchronous dataset replay: one ReplayStamp -> exactly one action chunk.

This is the sibling of ``serve_sync.py`` with the policy removed. The client
sends a ``ReplayStamp`` instead of an ``ObservationPacket``, and the answer is a
slice of a recorded trajectory instead of a forward pass. The pairing rule is
the same, only the field name changes: ``ActionMessage.source_seq_used`` must
equal ``ReplayStamp.stamp_seq``.

Two properties of this mode make the ``serve_sync`` machinery unnecessary:

* The whole episode is loaded into memory at startup, so answering a stamp is a
  numpy slice. There is no IO to move off the gRPC event loop, and answering
  inline is what makes "one stamp, exactly one message" true by construction.
* The cursor is derived from ``stamp_seq`` alone, so there is no state to reset
  between episodes and no way for two episodes to interleave.
"""

import json
import logging
import pathlib
import threading
from typing import Any

import numpy as np

from openpi.inin.serve import ACTION_MODE
from openpi.inin.serve import InferenceTransport

logger = logging.getLogger(__name__)

# The chunk column layout the client expects: xyz, quat_xyzw, gripper openness.
ACTION_DIM = 8
_QUAT_SLICE = slice(3, 7)


def load_episode_actions(data_root: pathlib.Path, repo_id: str, episode: int) -> np.ndarray:
    """Load one LeRobot episode's action column as a ``(T, 8)`` float32 array.

    The parquet is read directly rather than through ``LeRobotDataset`` because
    importing lerobot pulls in the whole torch stack and costs ~50 s, which is
    pure overhead for a server that loads no model. Reading only the ``action``
    column also skips decoding the PNG bytes stored alongside it.
    """
    root = pathlib.Path(data_root) / repo_id
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"{info_path} not found; is {root} a LeRobot v2.1 dataset?")
    info = json.loads(info_path.read_text(encoding="utf-8"))

    total_episodes = int(info["total_episodes"])
    if not 0 <= episode < total_episodes:
        raise ValueError(f"episode {episode} out of range for {repo_id}; expected 0 <= episode < {total_episodes}")

    chunks_size = int(info["chunks_size"])
    parquet_path = root / str(info["data_path"]).format(
        episode_chunk=episode // chunks_size,
        episode_index=episode,
    )
    if not parquet_path.exists():
        raise FileNotFoundError(f"episode {episode} parquet not found at {parquet_path}")

    import pyarrow.parquet as pq

    table = pq.read_table(parquet_path, columns=["action"])
    actions = np.asarray(np.stack(table.column("action").to_pylist()), dtype=np.float32)
    _validate_actions(actions, str(parquet_path))
    return actions


def _validate_actions(actions: np.ndarray, source: str) -> None:
    """Reject at startup what the client would reject mid-replay."""
    if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
        raise ValueError(f"{source}: expected action shape (T, {ACTION_DIM}), got {actions.shape}")
    if actions.shape[0] == 0:
        raise ValueError(f"{source}: episode has no frames")
    quat_norms = np.linalg.norm(actions[:, _QUAT_SLICE], axis=1)
    degenerate = np.flatnonzero(quat_norms == 0.0)
    if degenerate.size:
        raise ValueError(
            f"{source}: zero TCP quaternion at frame(s) {degenerate[:5].tolist()}; "
            "the client would reject this chunk as invalid_action"
        )


class ReplaySyncCallbacks:
    """inin-stream ServerCallbacks answering every replay stamp with one chunk."""

    def __init__(
        self,
        actions: np.ndarray,
        *,
        exec_steps: int = 10,
        start_frame: int = 0,
        end_frame: int | None = None,
    ) -> None:
        if exec_steps <= 0:
            raise ValueError("exec_steps must be positive")
        if start_frame < 0:
            raise ValueError("start_frame must be non-negative")
        actions = np.asarray(actions, dtype=np.float32)
        _validate_actions(actions, "replay actions")

        total_frames = actions.shape[0]
        self.start_frame = min(start_frame, total_frames)
        self.end_frame = total_frames if end_frame is None else min(end_frame, total_frames)
        if self.end_frame <= self.start_frame:
            # An empty range could only ever answer replay_complete, which is a
            # misconfiguration worth failing on rather than serving.
            raise ValueError(
                f"selected frame range [{self.start_frame},{self.end_frame}) is empty; "
                f"the episode has {total_frames} frames"
            )

        self._actions = np.ascontiguousarray(actions, dtype=np.float32)
        self._exec_steps = exec_steps

        self._server: InferenceTransport | None = None
        self._lock = threading.Lock()
        self._robot_id: str | None = None
        self._last_stamp_seq: int | None = None
        # Progress is per-episode because a connection may replay several times;
        # the public counters stay lifetime totals for the disconnect summary.
        self._episode_chunks = 0

        self.sent_chunks = 0
        self.sent_completions = 0

    @property
    def replay_frames(self) -> int:
        return self.end_frame - self.start_frame

    @property
    def total_rounds(self) -> int:
        """Number of stamps needed to replay the selected range."""
        return -(-self.replay_frames // self._exec_steps)

    def attach_stream_server(self, server: InferenceTransport) -> None:
        self._server = server

    def start(self) -> None:
        """Present for symmetry with the inference servers; nothing runs in the background."""

    def close(self) -> None:
        """Present for symmetry with the inference servers; nothing runs in the background."""

    # --- inin-stream ServerCallbacks interface -------------------------------

    def on_schema_accepted(self, hello: Any) -> None:
        # ReplayStamp carries no robot_id, so the handshake is the only place to
        # learn which robot send_action()/send_control() should target.
        with self._lock:
            self._robot_id = hello.robot_id
        logger.info(
            "replay client ready robot_id=%s run_id=%s mode=%s",
            hello.robot_id,
            hello.run_id,
            hello.run_mode,
        )

    def on_episode_start(self, event: Any) -> None:
        # The cursor is a pure function of stamp_seq, which the client restarts
        # at 0 for every episode, so there is nothing to reset here.
        with self._lock:
            self._last_stamp_seq = None
            self._episode_chunks = 0
        logger.info(
            "replay episode started: %s frames=%d rounds=%d",
            event.episode_id,
            self.replay_frames,
            self.total_rounds,
        )

    def on_observation(self, obs: Any) -> None:
        # A replay client uploads no observations; ignore any that arrive rather
        # than letting them disturb the request/response pairing.
        del obs

    def on_replay_stamp(self, stamp: Any) -> None:
        # Answered inline: the reply must be exactly one message per stamp, and
        # slicing a preloaded array cannot block the gRPC event loop.
        try:
            self._answer(stamp)
        except Exception:
            # Sending an error message would need a protocol the client does not
            # implement, so silence is the agreed failure mode: the client hits
            # its action timeout and aborts the episode itself.
            logger.exception("replay failed for stamp_seq=%d; sending nothing", stamp.stamp_seq)

    def on_episode_end(self, event: Any) -> None:
        logger.info("replay episode ended: %s reason=%s", event.episode_id, event.reason)

    def on_abort(self, event: Any) -> None:
        logger.warning("replay episode aborted: %s reason=%s", event.episode_id, event.reason)

    def on_client_disconnected(self, robot_id: str) -> None:
        logger.info(
            "replay client disconnected robot_id=%s chunks=%d completions=%d",
            robot_id,
            self.sent_chunks,
            self.sent_completions,
        )
        with self._lock:
            self._robot_id = None
            self._last_stamp_seq = None

    # --- replay ---------------------------------------------------------------

    def chunk_for(self, stamp_seq: int) -> np.ndarray:
        """Return the non-overlapping slice replayed for ``stamp_seq``.

        Chunks are contiguous and never overlap: the client executes each one to
        completion, so a training-style sliding window would walk the arm
        backwards at every round boundary. The tail chunk is returned short
        rather than padded.
        """
        if stamp_seq < 0:
            raise ValueError(f"stamp_seq must be non-negative, got {stamp_seq}")
        start = self.start_frame + stamp_seq * self._exec_steps
        if start >= self.end_frame:
            return self._actions[:0]
        return self._actions[start : min(start + self._exec_steps, self.end_frame)]

    def _answer(self, stamp: Any) -> None:
        server = self._server
        if server is None:
            raise RuntimeError("ReplaySyncCallbacks server was not attached")
        with self._lock:
            robot_id = self._robot_id
            previous = self._last_stamp_seq
            self._last_stamp_seq = stamp.stamp_seq
        if robot_id is None:
            raise RuntimeError(f"no robot_id yet for stamp_seq={stamp.stamp_seq}; handshake was not seen")
        if previous is not None and stamp.stamp_seq != previous + 1:
            # Still answered: dropping it would only cost the client a timeout.
            logger.warning(
                "stamp_seq jumped from %d to %d; the client's rounds are not contiguous",
                previous,
                stamp.stamp_seq,
            )

        chunk = self.chunk_for(stamp.stamp_seq)
        if chunk.shape[0] == 0:
            # Deliberately sent on the stamp *after* the last chunk: the client
            # sends its stamp before it starts waiting, so a completion bundled
            # with the final chunk would be consumed by the previous round.
            server.send_control(
                robot_id=robot_id,
                control_type="replay_complete",
                params={"episode_id": stamp.episode_id},
            )
            self.sent_completions += 1
            logger.info("replay complete at stamp_seq=%d episode=%s", stamp.stamp_seq, stamp.episode_id)
            return

        server.send_action(
            robot_id=robot_id,
            action=chunk,
            source_seq_used=stamp.stamp_seq,
            action_mode=ACTION_MODE,
            obs_stamp_ns=stamp.stamp_ns,
        )
        self.sent_chunks += 1
        with self._lock:
            self._episode_chunks += 1
            episode_chunks = self._episode_chunks
        logger.info(
            "chunk %d/%d for stamp_seq=%d shape=%s",
            episode_chunks,
            self.total_rounds,
            stamp.stamp_seq,
            chunk.shape,
        )

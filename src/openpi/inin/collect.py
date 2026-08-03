"""Data-collection callbacks bridging inin-stream to a LeRobot dataset.

Ported from inin-server's ``BCCollectionCallbacks``: a bounded work queue
feeds a background writer thread, episodes are deduplicated and committed
only when every observation sequence number in ``0..frames-1`` arrived, and
the workstation receives ``collection_episode_committed`` /
``collection_episode_failed`` / ``collection_episode_aborted`` controls.

Deduplication only covers retransmissions from one workstation session. An
abort or a write failure drops the attempt without blacklisting its
episode_id, and a reconnect with a new ``run_id`` resets the dedup state,
because in both cases the workstation reuses episode ids it already sent.
"""

import logging
import pathlib
import queue
import threading
from typing import Any, Protocol

from inin_stream.common.serialization import ObservationFrame
from inin_stream.dataset import DatasetEvent
from inin_stream.dataset import DatasetWriter
from inin_stream.dataset import DatasetWriterConfig
from inin_stream.dataset import build_dataset_writer
from inin_stream.schema import Schema

logger = logging.getLogger(__name__)


class CollectionTransport(Protocol):
    def send_control(self, robot_id: str, control_type: str, *, params: dict[str, str]) -> None: ...


class CollectionCallbacks:
    """inin-stream ServerCallbacks implementation that persists episodes."""

    def __init__(
        self,
        schema: Schema,
        data_root: pathlib.Path,
        repo_id: str,
        task: str | None = None,
        effective_config: dict[str, Any] | None = None,
    ) -> None:
        self.schema = schema
        self.data_root = pathlib.Path(data_root)
        self.repo_id = repo_id
        # When set, this instruction is written to every episode, overriding
        # any task carried in the workstation's episode_start metadata.
        self.task = task.strip() if task else None
        self.effective_config = effective_config or {}
        self.writer: DatasetWriter | None = None
        self._run_id = ""
        self._robot_id = ""
        self.observation_count = 0
        # Everything below is owned by the writer thread only.
        self._session_run_id = ""
        self._episode_sequences: dict[str, set[int]] = {}
        self._pending_end: dict[str, Any] = {}
        self._seen_events: set[tuple[str, str]] = set()
        self._active_episode: str | None = None
        # Episodes whose in-flight attempt was dropped by an abort or a write
        # failure. Trailing frames of that attempt are ignored until the
        # workstation re-records the id with a fresh episode_start.
        self._discarded: set[str] = set()
        self._work: queue.Queue[tuple[str, object] | None] = queue.Queue(maxsize=512)
        self._worker: threading.Thread | None = None
        self._server: CollectionTransport | None = None

    def attach_stream_server(self, server: CollectionTransport) -> None:
        self._server = server

    def close(self) -> None:
        if self._worker is not None:
            self._work.put(None)
            self._worker.join(timeout=30.0)
        if self.writer is not None:
            self.writer.close()

    # --- inin-stream ServerCallbacks interface -------------------------------

    def on_schema_accepted(self, hello: Any) -> None:
        self._run_id = hello.run_id
        self._robot_id = hello.robot_id
        if self.writer is None:
            self.writer = build_dataset_writer(
                DatasetWriterConfig(
                    collection_root=self.data_root,
                    dataset_format="lerobot",
                    schema=self.schema,
                    run_id=self._run_id,
                    robot_id=self._robot_id,
                    effective_config=self.effective_config,
                    repo_id=self.repo_id,
                )
            )
        if self._worker is None:
            self._worker = threading.Thread(target=self._worker_main, name="openpi-inin-collection-writer", daemon=True)
            self._worker.start()
        # Reset per-episode state on the writer thread rather than here: this
        # callback runs on the gRPC loop while the worker owns that state.
        self._enqueue("session", hello)
        logger.info("collection schema accepted robot_id=%s run_id=%s", hello.robot_id, hello.run_id)

    def on_episode_start(self, event: Any) -> None:
        self._enqueue("start", event)

    def on_observation(self, obs: ObservationFrame) -> None:
        self._enqueue("observation", obs)

    def on_episode_end(self, event: Any) -> None:
        self._enqueue("end", event)

    def on_abort(self, event: Any) -> None:
        self._enqueue("abort", event)

    def on_replay_stamp(self, stamp: Any) -> None:
        pass

    def on_client_disconnected(self, robot_id: str) -> None:
        # A disconnect is not a collection boundary; pending episode state is
        # kept so the same workstation process can retransmit idempotently
        # after reconnect. A reconnect carrying a new run_id restarts the
        # episode-id namespace and resets that state instead.
        logger.info("client disconnected; pending collection state retained: %s", robot_id)

    # --- background writer ----------------------------------------------------

    def _enqueue(self, kind: str, value: object) -> None:
        # Backpressure is deliberate: it reaches gRPC rather than silently
        # dropping an observation while disk / LeRobot is slow.
        self._work.put((kind, value))

    def _worker_main(self) -> None:
        while True:
            item = self._work.get()
            if item is None:
                return
            kind, value = item
            try:
                if kind == "session":
                    self._process_session(value)
                elif kind == "start":
                    self._process_start(value)
                elif kind == "observation":
                    self._process_observation(value)
                elif kind == "end":
                    self._process_end(value)
                else:
                    self._process_abort(value)
            except Exception as exc:  # report failures to the robot side
                episode_id = getattr(value, "episode_id", "")
                logger.exception("collection write failed episode=%s", episode_id)
                if episode_id:
                    # Drop the attempt rather than blacklisting the id, so the
                    # workstation can re-record it after the failure.
                    self._discard_episode(episode_id, reason="write_failed")
                self._send_control("collection_episode_failed", episode_id, error=str(exc))

    def _process_session(self, hello: Any) -> None:
        run_id = getattr(hello, "run_id", "")
        if run_id == self._session_run_id:
            return
        if self._session_run_id:
            logger.info(
                "collection session changed run_id=%s -> %s; resetting per-episode state",
                self._session_run_id,
                run_id,
            )
            if self._active_episode is not None:
                # The previous session died mid-episode. Its frames must not be
                # concatenated into the first episode of the new session.
                self._discard_episode(self._active_episode, reason="session_changed")
        self._session_run_id = run_id
        # A new workstation process restarts its episode-id counter, so the
        # previous session's ids must not deduplicate the new session's.
        self._seen_events.clear()
        self._episode_sequences.clear()
        self._pending_end.clear()
        self._discarded.clear()

    def _process_start(self, event: Any) -> None:
        key = (event.episode_id, "episode_start")
        if key in self._seen_events:
            # An idempotent retransmission after a reconnect. A re-record of an
            # aborted or failed episode is not one: dropping that attempt also
            # dropped this key, so the retry is treated as a fresh episode.
            logger.info("ignoring duplicate episode_start: %s", event.episode_id)
            return
        self._discarded.discard(event.episode_id)
        self._episode_sequences[event.episode_id] = set()
        self._writer().on_event(_dataset_event(event, task=self.task))
        self._active_episode = event.episode_id
        self._seen_events.add(key)
        logger.info("episode started: %s task=%r", event.episode_id, self.task or dict(event.metadata).get("task"))

    def _process_observation(self, obs: ObservationFrame) -> None:
        if obs.episode_id in self._discarded:
            return
        sequences = self._episode_sequences.get(obs.episode_id)
        if sequences is None:
            raise ValueError(f"observation received before episode_start: {obs.episode_id}")
        if obs.obs_seq in sequences:
            return
        self._writer().on_observation(obs)
        sequences.add(obs.obs_seq)
        self.observation_count += 1
        self._maybe_commit(obs.episode_id)

    def _process_end(self, event: Any) -> None:
        if event.episode_id in self._discarded:
            return
        self._pending_end[event.episode_id] = event
        self._maybe_commit(event.episode_id)

    def _maybe_commit(self, episode_id: str) -> None:
        event = self._pending_end.get(episode_id)
        if event is None:
            return
        try:
            expected = int(event.metadata["frames"])
        except (KeyError, ValueError) as exc:
            raise ValueError("episode_end requires integer metadata.frames") from exc
        sequences = self._episode_sequences.get(episode_id, set())
        if sequences != set(range(expected)):
            return
        key = (episode_id, "episode_end")
        if key in self._seen_events:
            return
        self._writer().on_event(_dataset_event(event))
        self._seen_events.add(key)
        self._pending_end.pop(episode_id, None)
        if self._active_episode == episode_id:
            self._active_episode = None
        self._send_control("collection_episode_committed", episode_id, frames=str(expected))
        logger.info("episode committed: %s frames=%d", episode_id, expected)

    def _process_abort(self, event: Any) -> None:
        if (event.episode_id, "episode_end") in self._seen_events:
            raise ValueError("cannot abort an already committed episode")
        self._discard_episode(event.episode_id, reason=event.reason or "abort")
        self._send_control("collection_episode_aborted", event.episode_id)
        logger.info("episode aborted: %s reason=%s", event.episode_id, event.reason)

    def _discard_episode(self, episode_id: str, reason: str) -> None:
        """Drop the in-flight attempt at ``episode_id`` and forget it ever started.

        The workstation numbers episodes by committed count, so re-recording
        after an abort reuses the same episode_id. Keeping the episode_start
        dedup key would make that retry look like a post-reconnect
        retransmission and silently discard the whole episode.
        """
        if (episode_id, "episode_end") in self._seen_events:
            # Already committed: nothing is in flight, and dropping its dedup
            # key would let a retransmission write the episode a second time.
            return
        if self._active_episode == episode_id:
            try:
                self._writer().on_event(DatasetEvent(event_type="abort", episode_id=episode_id, reason=reason))
            except Exception:
                logger.exception("writer failed to discard episode: %s", episode_id)
            self._active_episode = None
        self._discarded.add(episode_id)
        self._pending_end.pop(episode_id, None)
        self._episode_sequences.pop(episode_id, None)
        self._seen_events.discard((episode_id, "episode_start"))

    def _send_control(self, control_type: str, episode_id: str, **params: str) -> None:
        if self._server is None:
            return
        self._server.send_control(self._robot_id, control_type, params={"episode_id": episode_id, **params})

    def _writer(self) -> DatasetWriter:
        if self.writer is None:
            raise RuntimeError("collection writer is not initialized; schema was not accepted")
        return self.writer


def _dataset_event(event: Any, task: str | None = None) -> DatasetEvent:
    metadata = dict(event.metadata)
    if task:
        metadata["task"] = task
    return DatasetEvent(
        event_type=event.event_type,
        episode_id=event.episode_id,
        reason=event.reason,
        metadata=metadata,
    )

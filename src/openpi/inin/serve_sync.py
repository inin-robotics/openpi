"""Synchronous online inference: one observation -> exactly one action chunk.

The rate-limited pipeline in ``serve.py`` keeps a latest-only slot and drops
stale observations, which is right when the robot executes chunks continuously.
This mode instead treats every ObservationPacket as a request and answers it
with exactly one ActionMessage, so the robot can finish a whole chunk before
publishing its next observation. Two consequences follow:

* The intake queue never discards. A dropped frame would leave the client
  waiting out its whole action timeout, since nothing else will answer it.
* Inference still runs on a worker thread. inin-stream invokes callbacks
  synchronously on the gRPC asyncio loop, so blocking in ``on_observation``
  would also delay the ``episode_end`` / ``abort`` that arrive on that loop.
"""

import logging
import queue
import threading
import time
from typing import Any

from inin_stream.common.serialization import ObservationFrame

from openpi.inin.serve import ACTION_MODE
from openpi.inin.serve import InferenceTransport
from openpi.inin.serve import _Policy
from openpi.inin.serve import frame_to_observation
from openpi.inin.serve import infer_chunk

logger = logging.getLogger(__name__)


class SyncInferenceCallbacks:
    """inin-stream ServerCallbacks answering every observation with one chunk."""

    def __init__(
        self,
        policy: _Policy,
        *,
        prompt: str,
        exec_steps: int = 10,
    ) -> None:
        if exec_steps <= 0:
            raise ValueError("exec_steps must be positive")
        if not prompt.strip():
            raise ValueError("prompt must be a non-empty instruction")
        self._policy = policy
        self._prompt = prompt
        self._exec_steps = exec_steps

        self._server: InferenceTransport | None = None
        self._lock = threading.Lock()
        self._robot_id: str | None = None
        self._episode_id: str | None = None

        self._pending: queue.Queue[ObservationFrame] = queue.Queue()
        self._stop = threading.Event()
        self._worker: threading.Thread | None = None
        self._backlog_warned = False
        self._short_chunk_warned = False

        self.observation_count = 0
        self.sent_chunks = 0

    def attach_stream_server(self, server: InferenceTransport) -> None:
        self._server = server

    def start(self) -> None:
        if self._worker is None:
            self._worker = threading.Thread(target=self._worker_main, name="openpi-inin-inference-sync", daemon=True)
            self._worker.start()

    def close(self) -> None:
        self._stop.set()
        if self._worker is not None:
            self._worker.join(timeout=10.0)

    # --- inin-stream ServerCallbacks interface -------------------------------

    def on_schema_accepted(self, hello: Any) -> None:
        with self._lock:
            self._robot_id = hello.robot_id
        logger.info(
            "sync inference client ready robot_id=%s run_id=%s mode=%s",
            hello.robot_id,
            hello.run_id,
            hello.run_mode,
        )

    def on_episode_start(self, event: Any) -> None:
        # The prompt is fixed at startup and never taken from the robot: its
        # episode_start metadata has been seen carrying the run mode name rather
        # than the instruction, which silently degrades a language-conditioned
        # policy. The metadata task is still logged so a client-side mismatch
        # stays visible.
        metadata_task = dict(event.metadata).get("task", "").strip()
        if metadata_task and metadata_task != self._prompt:
            logger.warning("client episode_start task %r ignored; serving prompt %r", metadata_task, self._prompt)
        with self._lock:
            self._episode_id = event.episode_id
        self._drain_pending()
        logger.info("sync inference episode started: %s prompt=%r", event.episode_id, self._prompt)

    def on_observation(self, obs: ObservationFrame) -> None:
        with self._lock:
            if self._episode_id != obs.episode_id:
                return
            self.observation_count += 1
        self._pending.put(obs)
        # Under the synchronous contract the client publishes its next
        # observation only after executing the previous chunk, so anything
        # deeper than one pending frame means that contract was broken.
        if self._pending.qsize() > 1 and not self._backlog_warned:
            self._backlog_warned = True
            logger.warning(
                "observation backlog=%d: client is publishing without waiting for the previous chunk",
                self._pending.qsize(),
            )

    def on_episode_end(self, event: Any) -> None:
        self._finish_episode(event, "ended")

    def on_abort(self, event: Any) -> None:
        self._finish_episode(event, "aborted")

    def on_replay_stamp(self, stamp: Any) -> None:
        pass

    def on_client_disconnected(self, robot_id: str) -> None:
        logger.info(
            "sync inference client disconnected robot_id=%s observations=%d chunks=%d",
            robot_id,
            self.observation_count,
            self.sent_chunks,
        )
        with self._lock:
            self._robot_id = None
            self._episode_id = None
        self._drain_pending()

    def _finish_episode(self, event: Any, status: str) -> None:
        with self._lock:
            if self._episode_id == event.episode_id:
                self._episode_id = None
        self._drain_pending()
        logger.info("sync inference episode %s: %s reason=%s", status, event.episode_id, event.reason)

    def _drain_pending(self) -> None:
        """Discard frames of an episode that is no longer active."""
        while True:
            try:
                self._pending.get_nowait()
            except queue.Empty:
                return

    # --- inference worker -----------------------------------------------------

    def _worker_main(self) -> None:
        while not self._stop.is_set():
            try:
                frame = self._pending.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                self._infer_and_send(frame)
            except Exception:
                # The client has no other signal that this frame failed; it
                # falls back to its action timeout and decides whether to
                # republish or abort.
                logger.exception("sync inference failed for obs_seq=%d", frame.obs_seq)

    def _build_policy_observation(self, frame: ObservationFrame, task: str) -> dict:
        """Build one policy input; subclasses may replace live data with another source."""
        return frame_to_observation(frame, task)

    def _send_prediction(
        self,
        *,
        robot_id: str,
        frame: ObservationFrame,
        observation: dict,
        chunk: Any,
    ) -> bool:
        """Send a prediction and return whether an action was emitted."""
        server = self._server
        if server is None:
            raise RuntimeError("SyncInferenceCallbacks server was not attached")
        server.send_action(
            robot_id=robot_id,
            action=chunk,
            source_seq_used=frame.obs_seq,
            action_mode=ACTION_MODE,
            obs_stamp_ns=frame.stamp_ns,
        )
        return True

    def _infer_and_send(self, frame: ObservationFrame) -> None:
        with self._lock:
            robot_id = self._robot_id
            episode_id = self._episode_id
        if robot_id is None or episode_id != frame.episode_id:
            logger.info("dropping obs_seq=%d from inactive episode %s", frame.obs_seq, frame.episode_id)
            return

        start = time.monotonic()
        observation = self._build_policy_observation(frame, self._prompt)
        chunk = infer_chunk(self._policy, observation)[: self._exec_steps]
        elapsed_ms = (time.monotonic() - start) * 1e3
        if chunk.shape[0] < self._exec_steps and not self._short_chunk_warned:
            self._short_chunk_warned = True
            logger.warning(
                "policy action horizon yields %d steps, fewer than exec_steps=%d; sending what it produced",
                chunk.shape[0],
                self._exec_steps,
            )

        with self._lock:
            if self._episode_id != frame.episode_id:
                logger.info(
                    "episode %s ended during inference; chunk for obs_seq=%d dropped", frame.episode_id, frame.obs_seq
                )
                return
        sent = self._send_prediction(
            robot_id=robot_id,
            frame=frame,
            observation=observation,
            chunk=chunk,
        )
        if sent:
            self.sent_chunks += 1
        # This latency is the robot's idle time between chunks, so it is the
        # number to watch when tuning exec_steps.
        if sent:
            logger.info(
                "chunk %d for obs_seq=%d shape=%s inference=%.0f ms",
                self.sent_chunks,
                frame.obs_seq,
                chunk.shape,
                elapsed_ms,
            )

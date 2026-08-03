"""Online inference: inin-stream observations -> openpi policy -> action chunks.

Unlike inin-server's original implementation, inference does not run inside
the gRPC callback (which would block both observation intake and action
sending for the ~100ms of a pi0 forward pass). Observations land in a
latest-only slot and a dedicated worker thread runs the policy, so the stream
stays responsive and stale observations are dropped instead of queueing.
"""

import io
import logging
import threading
import time
from typing import Any, Protocol

from inin_stream.common.serialization import ObservationFrame
import numpy as np

from openpi.inin import conversions
from openpi.policies import inin_policy

logger = logging.getLogger(__name__)

ACTION_MODE = "absolute_tcp_gripper_chunk"


class InferenceTransport(Protocol):
    def send_action(
        self,
        robot_id: str,
        action: np.ndarray,
        source_seq_used: int,
        action_mode: str,
        obs_stamp_ns: int | None = None,
    ) -> None: ...

    def send_control(
        self,
        robot_id: str,
        control_type: str,
        message: str = "",
        params: dict[str, str] | None = None,
    ) -> None: ...


class _Policy(Protocol):
    def infer(self, obs: dict) -> dict: ...


def frame_to_observation(frame: ObservationFrame, prompt: str) -> dict:
    """Build the openpi observation dict (post-repack keys) from a stream frame."""
    state = np.concatenate(
        [
            np.asarray(frame.tensors["robot.joint_pos"], dtype=np.float32),
            np.asarray(frame.tensors["robot.tcp_pose"], dtype=np.float32),
            np.asarray(frame.tensors["robot.gripper_openness"], dtype=np.float32),
        ]
    )
    return {
        "state": state,
        "image": _decode_image_rgb(frame.images["camera.base.rgb"]),
        "wrist_image": _decode_image_rgb(frame.images["camera.wrist.rgb"]),
        "prompt": prompt,
    }


def _decode_image_rgb(data: bytes) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"), dtype=np.uint8)


def infer_chunk(policy: _Policy, obs: dict) -> np.ndarray:
    """Run the policy and convert its rotvec chunk to the on-the-wire quat chunk."""
    result = policy.infer(obs)
    actions = np.asarray(result["actions"])
    if actions.ndim != 2 or actions.shape[1] != conversions.ACTION_ROTVEC_DIM:
        raise ValueError(f"unexpected policy action shape: {actions.shape}")
    return conversions.rotvec_chunk_to_quat_chunk(actions)


def warmup(policy: _Policy, *, prompt: str, iterations: int = 2) -> None:
    """Run the policy on synthetic frames before the stream server accepts clients.

    Otherwise the robot's first observation pays for cuDNN autotuning, the first
    PaliGemma forward and the jax.jit trace inside resize_with_pad, and a GPU OOM
    there would surface as a swallowed exception in the inference worker rather
    than a startup failure. ``make_inin_example`` keeps the raw 288x384 camera
    resolution, so resize_with_pad compiles for the shape the robot really sends;
    a pre-resized example would leave that compile for the first real frame.
    """
    for i in range(1, iterations + 1):
        example = inin_policy.make_inin_example()
        example["prompt"] = prompt
        start = time.monotonic()
        chunk = infer_chunk(policy, example)
        logger.info(
            "warmup %d/%d took %.0f ms (chunk shape=%s)",
            i,
            iterations,
            (time.monotonic() - start) * 1e3,
            chunk.shape,
        )


class InferenceCallbacks:
    """inin-stream ServerCallbacks implementation driving an openpi policy."""

    def __init__(
        self,
        policy: _Policy,
        *,
        prompt: str,
        rate_hz: float = 5.0,
    ) -> None:
        if rate_hz <= 0.0:
            raise ValueError("rate_hz must be positive")
        if not prompt.strip():
            raise ValueError("prompt must be a non-empty instruction")
        self._policy = policy
        self._prompt = prompt
        self._period_s = 1.0 / rate_hz

        self._server: InferenceTransport | None = None
        self._robot_id: str | None = None
        self._episode_id: str | None = None

        self._cv = threading.Condition()
        self._latest: ObservationFrame | None = None
        self._stop = threading.Event()
        self._worker: threading.Thread | None = None
        # Earliest monotonic time the next forward pass may start.
        self._next_infer_s: float | None = None

        self.observation_count = 0
        self.sent_chunks = 0

    def attach_stream_server(self, server: InferenceTransport) -> None:
        self._server = server

    def start(self) -> None:
        if self._worker is None:
            self._worker = threading.Thread(target=self._worker_main, name="openpi-inin-inference", daemon=True)
            self._worker.start()

    def close(self) -> None:
        self._stop.set()
        with self._cv:
            self._cv.notify_all()
        if self._worker is not None:
            self._worker.join(timeout=10.0)

    # --- inin-stream ServerCallbacks interface -------------------------------

    def on_schema_accepted(self, hello: Any) -> None:
        self._robot_id = hello.robot_id
        logger.info("inference client ready robot_id=%s run_id=%s", hello.robot_id, hello.run_id)

    def on_episode_start(self, event: Any) -> None:
        # The prompt is fixed at startup and never taken from the robot: its
        # episode_start metadata has been seen carrying the run mode name rather
        # than the instruction, which silently degrades a language-conditioned
        # policy. The metadata task is still logged so a client-side mismatch
        # stays visible.
        metadata_task = dict(event.metadata).get("task", "").strip()
        if metadata_task and metadata_task != self._prompt:
            logger.warning("client episode_start task %r ignored; serving prompt %r", metadata_task, self._prompt)
        with self._cv:
            self._episode_id = event.episode_id
            self._latest = None
            self._next_infer_s = None
        logger.info("inference episode started: %s prompt=%r", event.episode_id, self._prompt)

    def on_observation(self, obs: ObservationFrame) -> None:
        with self._cv:
            if self._episode_id != obs.episode_id:
                return
            self._latest = obs
            self.observation_count += 1
            self._cv.notify()

    def on_episode_end(self, event: Any) -> None:
        self._finish_episode(event, "ended")

    def on_abort(self, event: Any) -> None:
        self._finish_episode(event, "aborted")

    def on_replay_stamp(self, stamp: Any) -> None:
        pass

    def on_client_disconnected(self, robot_id: str) -> None:
        logger.info(
            "inference client disconnected robot_id=%s observations=%d chunks=%d",
            robot_id,
            self.observation_count,
            self.sent_chunks,
        )
        with self._cv:
            self._robot_id = None
            self._episode_id = None
            self._latest = None

    def _finish_episode(self, event: Any, status: str) -> None:
        with self._cv:
            if self._episode_id == event.episode_id:
                self._episode_id = None
                self._latest = None
        logger.info("inference episode %s: %s reason=%s", status, event.episode_id, event.reason)

    # --- inference worker -----------------------------------------------------

    def _worker_main(self) -> None:
        while not self._stop.is_set():
            if not self._wait_for_next_slot():
                return
            # Sleeping before the slot's observation is read keeps the frame as
            # fresh as the stream allows, which bounds how stale the TCP anchor
            # of the emitted chunk is.
            frame = self._take_latest(timeout_s=0.2)
            if frame is None:
                continue
            with self._cv:
                self._next_infer_s = time.monotonic() + self._period_s
            try:
                self._infer_and_send(frame)
            except Exception:  # keep serving after a bad frame
                logger.exception("inference failed for obs_seq=%d", frame.obs_seq)

    def _wait_for_next_slot(self) -> bool:
        """Sleep until the next inference slot; False when shutting down.

        The slot is scheduled from the start of the previous forward pass, not
        from the moment its chunk was sent: rate-limiting after the send would
        stretch the real interval to ``period + inference_time`` and silently
        serve well below ``rate_hz``.
        """
        with self._cv:
            deadline = self._next_infer_s
        if deadline is None:
            return True
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return True
        return not self._stop.wait(remaining)

    def _take_latest(self, timeout_s: float) -> ObservationFrame | None:
        with self._cv:
            if self._latest is None and timeout_s > 0:
                self._cv.wait(timeout=timeout_s)
            frame = self._latest
            self._latest = None
            return frame

    def _infer_and_send(self, frame: ObservationFrame) -> None:
        with self._cv:
            robot_id = self._robot_id
        if robot_id is None:
            return
        chunk = infer_chunk(self._policy, frame_to_observation(frame, self._prompt))
        server = self._server
        if server is None:
            raise RuntimeError("InferenceCallbacks server was not attached")
        server.send_action(
            robot_id=robot_id,
            action=chunk,
            source_seq_used=frame.obs_seq,
            action_mode=ACTION_MODE,
            obs_stamp_ns=frame.stamp_ns,
        )
        self.sent_chunks += 1
        if self.sent_chunks == 1 or self.sent_chunks % 30 == 0:
            logger.info(
                "inference progress chunks=%d observations=%d latest_obs_seq=%d shape=%s",
                self.sent_chunks,
                self.observation_count,
                frame.obs_seq,
                chunk.shape,
            )

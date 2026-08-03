"""Dataset-driven open-loop inference on top of the synchronous inin server."""

from collections.abc import Callable, Mapping, Sequence
import dataclasses
import logging
import math
import pathlib
from typing import Any

from inin_stream.common.serialization import ObservationFrame
import numpy as np

from openpi import transforms as _transforms
from openpi.inin.serve import _Policy
from openpi.inin.serve_sync import SyncInferenceCallbacks
from openpi.training import config as _train_config

logger = logging.getLogger(__name__)

# Each dataset frame carries its own prompt (PromptFromLeRobotTask), and
# _build_policy_observation replaces the whole observation, so the base class
# prompt is never reached.
_UNUSED_PROMPT = "dataset open-loop"


class DatasetExhaustedError(Exception):
    """Raised when an episode source has no frame left to replay."""


@dataclasses.dataclass(frozen=True)
class ReplayFrame:
    dataset_index: int
    frame_index: int
    observation: dict

    @property
    def prompt(self) -> str:
        return str(self.observation["prompt"])

    @property
    def state(self) -> np.ndarray:
        return np.asarray(self.observation["state"], dtype=np.float32)


class LeRobotEpisodeSource:
    """Expose post-repack policy observations from one LeRobot episode."""

    def __init__(
        self,
        dataset: Sequence[Mapping[str, Any]],
        transform: Callable[[dict], dict],
        *,
        start_frame: int = 0,
        end_frame: int | None = None,
        frame_step: int = 1,
    ) -> None:
        if frame_step <= 0:
            raise ValueError("frame_step must be positive")
        if start_frame < 0:
            raise ValueError("start_frame must be non-negative")
        dataset_end = len(dataset)
        selected_end = dataset_end if end_frame is None else min(end_frame, dataset_end)
        if selected_end < start_frame:
            raise ValueError(f"end_frame ({selected_end}) must be >= start_frame ({start_frame})")

        self._dataset = dataset
        self._transform = transform
        self.start_frame = start_frame
        self.end_frame = selected_end
        self.frame_step = frame_step
        self._index = start_frame

    @classmethod
    def from_train_config(
        cls,
        train_config: _train_config.TrainConfig,
        *,
        episode: int,
        data_root: pathlib.Path | None,
        repo_id: str | None = None,
        start_frame: int = 0,
        end_frame: int | None = None,
        frame_step: int = 1,
    ) -> "LeRobotEpisodeSource":
        from lerobot.common.datasets import lerobot_dataset

        data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
        selected_repo = repo_id or data_config.repo_id
        if selected_repo is None:
            raise ValueError(f"config {train_config.name!r} has no repo_id; pass --repo-id")
        root = str(data_root / selected_repo) if data_root is not None else None
        metadata = lerobot_dataset.LeRobotDatasetMetadata(selected_repo, root=root)
        if not 0 <= episode < metadata.total_episodes:
            raise ValueError(
                f"episode {episode} out of range for {selected_repo}; expected 0 <= episode < {metadata.total_episodes}"
            )
        dataset = lerobot_dataset.LeRobotDataset(selected_repo, root=root, episodes=[episode])
        transform = _transforms.compose(
            [
                _transforms.PromptFromLeRobotTask(metadata.tasks),
                *data_config.repack_transforms.inputs,
            ]
        )
        source = cls(
            dataset,
            transform,
            start_frame=start_frame,
            end_frame=end_frame,
            frame_step=frame_step,
        )
        logger.info(
            "loaded open-loop source repo=%s episode=%d frames=%d range=[%d,%d) step=%d",
            selected_repo,
            episode,
            len(dataset),
            source.start_frame,
            source.end_frame,
            source.frame_step,
        )
        return source

    @property
    def index(self) -> int:
        return self._index

    def reset(self) -> None:
        self._index = self.start_frame

    def current(self) -> ReplayFrame:
        if self._index >= self.end_frame:
            raise DatasetExhaustedError(
                f"dataset replay exhausted at index {self._index}; selected range ends at {self.end_frame}"
            )
        item = dict(self._dataset[self._index])
        observation = dict(self._transform(item))
        # The policy inference path must not receive the dataset supervision.
        observation.pop("actions", None)
        required = {"state", "image", "wrist_image", "prompt"}
        missing = required.difference(observation)
        if missing:
            raise KeyError(f"dataset frame {self._index} is missing policy inputs: {sorted(missing)}")
        frame_index = int(np.asarray(item.get("frame_index", self._index)).item())
        return ReplayFrame(
            dataset_index=self._index,
            frame_index=frame_index,
            observation=observation,
        )

    def advance(self) -> None:
        self._index += self.frame_step


def pose_error(live_tcp_pose: np.ndarray, dataset_state: np.ndarray) -> tuple[float, float]:
    """Return TCP translation error in metres and rotation error in degrees."""
    live = np.asarray(live_tcp_pose, dtype=np.float64)
    state = np.asarray(dataset_state, dtype=np.float64)
    if live.shape != (7,):
        raise ValueError(f"expected live tcp pose shape (7,), got {live.shape}")
    if state.shape != (14,):
        raise ValueError(f"expected dataset state shape (14,), got {state.shape}")
    translation_m = float(np.linalg.norm(live[:3] - state[6:9]))
    live_quat = live[3:7]
    dataset_quat = state[9:13]
    live_norm = float(np.linalg.norm(live_quat))
    dataset_norm = float(np.linalg.norm(dataset_quat))
    if live_norm == 0.0 or dataset_norm == 0.0:
        raise ValueError("TCP quaternion norm must be positive")
    cosine = abs(float(np.dot(live_quat / live_norm, dataset_quat / dataset_norm)))
    rotation_deg = math.degrees(2.0 * math.acos(np.clip(cosine, -1.0, 1.0)))
    return translation_m, rotation_deg


class OpenLoopInferenceCallbacks(SyncInferenceCallbacks):
    """Use dataset observations for inference while retaining sync transport."""

    def __init__(
        self,
        policy: _Policy,
        source: LeRobotEpisodeSource,
        *,
        exec_steps: int = 10,
        execute: bool = False,
        max_translation_m: float = 0.03,
        max_rotation_deg: float = 15.0,
    ) -> None:
        if max_translation_m < 0.0:
            raise ValueError("max_translation_m must be non-negative")
        if max_rotation_deg < 0.0:
            raise ValueError("max_rotation_deg must be non-negative")
        super().__init__(
            policy,
            prompt=_UNUSED_PROMPT,
            exec_steps=exec_steps,
        )
        self._source = source
        self._execute = execute
        self._max_translation_m = max_translation_m
        self._max_rotation_deg = max_rotation_deg
        self._replay_frame: ReplayFrame | None = None
        self._halted = False
        self._stop_sent = False

    def on_episode_start(self, event: Any) -> None:
        self._source.reset()
        self._replay_frame = None
        self._halted = False
        self._stop_sent = False
        super().on_episode_start(event)
        logger.info(
            "open-loop replay reset episode=%s dataset_index=%d execute=%s",
            event.episode_id,
            self._source.index,
            self._execute,
        )

    def _build_policy_observation(self, frame: ObservationFrame, task: str) -> dict:
        del task
        if self._halted:
            raise DatasetExhaustedError("open-loop replay is halted")
        self._replay_frame = self._source.current()
        logger.info(
            "open-loop infer obs_seq=%d dataset_index=%d frame_index=%d prompt=%r",
            frame.obs_seq,
            self._replay_frame.dataset_index,
            self._replay_frame.frame_index,
            self._replay_frame.prompt,
        )
        return self._replay_frame.observation

    def _infer_and_send(self, frame: ObservationFrame) -> None:
        try:
            super()._infer_and_send(frame)
        except DatasetExhaustedError as error:
            self._halt(str(error), frame)

    def _send_prediction(
        self,
        *,
        robot_id: str,
        frame: ObservationFrame,
        observation: dict,
        chunk: Any,
    ) -> bool:
        replay_frame = self._replay_frame
        if replay_frame is None:
            raise RuntimeError("dataset replay frame was not prepared")
        translation_m, rotation_deg = pose_error(
            np.asarray(frame.tensors["robot.tcp_pose"]),
            replay_frame.state,
        )
        if translation_m > self._max_translation_m or rotation_deg > self._max_rotation_deg:
            self._halt(
                "pose mismatch: "
                f"translation={translation_m:.4f}m (limit={self._max_translation_m:.4f}m), "
                f"rotation={rotation_deg:.2f}deg (limit={self._max_rotation_deg:.2f}deg)",
                frame,
            )
            return False

        if not self._execute:
            logger.warning(
                "DRY-RUN: not sending dataset_index=%d chunk=%s "
                "(pose error %.4fm / %.2fdeg); pass --execute to enable motion",
                replay_frame.dataset_index,
                np.asarray(chunk).shape,
                translation_m,
                rotation_deg,
            )
            self._source.advance()
            return False

        sent = super()._send_prediction(
            robot_id=robot_id,
            frame=frame,
            observation=observation,
            chunk=chunk,
        )
        if sent:
            logger.info(
                "open-loop sent dataset_index=%d chunk=%s pose_error=%.4fm/%.2fdeg next_index=%d",
                replay_frame.dataset_index,
                np.asarray(chunk).shape,
                translation_m,
                rotation_deg,
                self._source.index + self._source.frame_step,
            )
            self._source.advance()
        return sent

    def _halt(self, reason: str, frame: ObservationFrame) -> None:
        self._halted = True
        logger.error("open-loop stopped at dataset_index=%d: %s", self._source.index, reason)
        if self._stop_sent:
            return
        server = self._server
        robot_id = self._robot_id
        if server is not None and robot_id is not None:
            server.send_control(
                robot_id=robot_id,
                control_type="stop_remote_action",
                message=reason,
                params={
                    "dataset_index": str(self._source.index),
                    "source_seq_used": str(frame.obs_seq),
                },
            )
            self._stop_sent = True

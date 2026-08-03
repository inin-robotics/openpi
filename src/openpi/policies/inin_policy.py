"""Transforms for the inin UR5 platform (bc-ur5-v2 schema).

The LeRobot dataset written by ``openpi.inin.collect`` stores the raw schema
representation (14D state with a quaternion TCP pose, 8D absolute
xyz+quaternion+gripper actions). These transforms convert to the 7D
xyz+rotation-vector+gripper representation used by the model, so quaternions
never reach ``DeltaActions`` / norm stats. See ``openpi.inin.conversions``.
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.inin import conversions
from openpi.models import model as _model


def make_inin_example() -> dict:
    """Creates a random input example for the inin UR5 policy."""
    quat = np.array([0.0, 0.0, 0.0, 1.0])
    state = np.concatenate([np.random.rand(6), np.random.rand(3), quat, np.random.rand(1)])
    return {
        "state": state.astype(np.float32),
        "image": np.random.randint(256, size=(288, 384, 3), dtype=np.uint8),
        "wrist_image": np.random.randint(256, size=(288, 384, 3), dtype=np.uint8),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class IninInputs(transforms.DataTransformFn):
    """Maps the raw inin representation to model inputs (training + inference)."""

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        state = conversions.state14_to_state7(np.asarray(data["state"]))

        base_image = _parse_image(data["image"])
        wrist_image = _parse_image(data["wrist_image"])

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                # The UR5 has no right wrist camera; pad with zeros.
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                # Padding images are masked out except for pi0-FAST.
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        # Actions are only present during training: an (action_horizon, 8)
        # absolute xyz+quat+gripper chunk from the dataset. Convert to
        # rotation vectors anchored at the state so DeltaActions stays sane.
        if "actions" in data:
            inputs["actions"] = conversions.quat_action_chunk_to_rotvec_chunk(
                np.asarray(data["actions"]), anchor_rotvec=state[3:6]
            )

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class IninOutputs(transforms.DataTransformFn):
    """Slices the model's padded action output back to the 7D UR5 action."""

    def __call__(self, data: dict) -> dict:
        # [x, y, z, rx, ry, rz, gripper]; the caller (openpi.inin.serve)
        # converts rotation vectors back to quaternions for the wire format.
        return {"actions": np.asarray(data["actions"][:, :7])}

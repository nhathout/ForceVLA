import dataclasses
from typing import ClassVar
import einops
import numpy as np
from openpi import transforms  
from openpi.models import model as _model

def make_forcevla_example() -> dict:
    """Creates a random input example compatible with Flexiv config."""
    return {
        "state": np.ones((13,), dtype=np.float32),  # 7 proprio dims + 6 force dims
        "image": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8), 
        "wrist_image": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8), 
        "prompt": "do something",
    }

# The six wrench channels of the 13-D UR5e state: [0:7] pose + gripper, [7:13] wrench.
WRENCH = slice(7, 13)


def wrench_history(w_now, w_lagged) -> np.ndarray:
    """M4: [w(t) - w(t-k) for each lagged wrench w(t-k)] -> (..., 6 * len(w_lagged)).

    The ONLY definition of the history: the training transform (ForceWindow), the offline
    twin (build_state_with_history), scripts/data/make_arm_norm_stats.py and any evaluator
    or rollout all call this. Works on one frame (6,) or on a batch of frames (N, 6).
    """
    return np.concatenate(
        [np.asarray(w_now) - np.asarray(w) for w in w_lagged], axis=-1
    ).astype(np.float32)


def build_state_with_history(states, t, ep_start, lags) -> np.ndarray:
    """Offline twin of ForceWindow for an (N, 13) array of states indexed like the dataset.

    `t` and `ep_start` index `states`. Lagged frames clamp to the episode's first frame,
    exactly as LeRobot clamps delta_timestamps (lerobot_dataset.py:665-678).
    """
    now = np.asarray(states[t])
    lagged = [np.asarray(states[max(ep_start, t - k)])[WRENCH] for k in lags]
    return np.concatenate([now[:13], wrench_history(now[WRENCH], lagged)]).astype(np.float32)


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class Forcevla_inputs(transforms.DataTransformFn):
    """
    This class is used to convert inputs to the model to the expected format. It is used for both training and inference.
    For your own dataset, you can copy this class and modify the keys based on the comments below to pipe
    the correct elements of your dataset into the model.
    """

    # The action dimension of the model. Will be used to pad state and actions for pi0 model (not pi0-FAST).
    # Do not change this for your own dataset.
    action_dim: int

    # Determines which model will be used.
    # Do not change this for your own dataset.
    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:
        # We only mask padding for pi0 model, not pi0-FAST. Do not change this for your own dataset.
        mask_padding = self.model_type == _model.ModelType.PI0

        # We pad the proprioceptive input to the action dimension of the model.
        # For pi0-FAST, we don't pad the state. For Libero, we don't need to differentiate
        # since the pi0-FAST action_dim = 7, which is < state_dim = 8, so pad is skipped.
        # Keep this for your own dataset, but if your dataset stores the proprioceptive input
        # in a different key than "observation/state", you should change it below.
        state = transforms.pad_to_dim(data["state"], self.action_dim)

        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference.
        # Keep this for your own dataset, but if your dataset stores the images
        # in a different key than "observation/image" or "observation/wrist_image",
        # you should change it below.
        # Pi0 models support three image inputs at the moment: one third-person view,
        # and two wrist views (left and right). If your dataset does not have a particular type
        # of image, e.g. wrist images, you can comment it out here and replace it with zeros like we do for the
        # right wrist image below.
        base_image = _parse_image(data["image"])
        left_wrist_image = _parse_image(data["wrist_image"])

        # Create inputs dict. Do not change the keys in the dict below.
        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": left_wrist_image,
                # Pad any non-existent images with zero-arrays of the appropriate shape.
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                # Mask any non-existent images with False (if ``mask_padding`` is True).
                "right_wrist_0_rgb": np.False_ if mask_padding else np.True_,
            },
        }

        # Pad actions to the model action dimension. Keep this for your own dataset.
        # Actions are only available during training.
        if "actions" in data:
            # We are padding to the model action dim.
            # For pi0-FAST, this is a no-op (since action_dim = 7).
            actions = transforms.pad_to_dim(data["actions"], self.action_dim)
            inputs["actions"] = actions

        # Pass the prompt (aka language instruction) to the model.
        # Keep this for your own dataset (but modify the key if the instruction is not
        # stored in "prompt"; the output dict always needs to have the key "prompt").
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs
    
@dataclasses.dataclass(frozen=True)
class ForceWindow(transforms.DataTransformFn):
    """M4/M8 (docs/plan_architecture_experiments.md sec 3-4). Runs BEFORE Forcevla_inputs.

    Training: the data loader fetches `state` at the frame offsets `delta_indices`
    (DataConfig.state_delta_indices), so `state` arrives as [len(delta_indices), 13]. This
    turns it into the model's state - state(t)[:13], plus (M4) the wrench-difference
    history - and (M8) replaces action dims 7:13 with the wrench at t+1..t+H.

    Inference: a 1-D state passes through; the caller has already built it. With
    history_lags set it MUST already carry the history, else Forcevla_inputs would
    zero-pad state[13:31] and the model would silently run without it (serve_policy,
    dump_policy_actions and eval_expert_routing all build 13-D), so that raises.
    """

    delta_indices: tuple[int, ...]
    history_lags: tuple[int, ...] = ()
    future_wrench: bool = False
    action_horizon: int = 50

    def __call__(self, data: dict) -> dict:
        s = np.asarray(data["state"])
        if s.ndim == 1:
            want = 13 + 6 * len(self.history_lags)
            if self.history_lags and s.shape[0] != want:
                raise ValueError(
                    f"ForceWindow: 1-D state has {s.shape[0]} dims, M4 (history_lags="
                    f"{self.history_lags}) needs {want}; build it with build_state_with_history")
            return data
        if s.ndim != 2 or s.shape[0] != len(self.delta_indices):
            raise ValueError(
                f"ForceWindow: state window has shape {s.shape}, expected "
                f"({len(self.delta_indices)}, 13) for delta_indices={self.delta_indices}")
        pos = {d: i for i, d in enumerate(self.delta_indices)}
        now = s[pos[0]]
        state = now[:13]
        if self.history_lags:
            state = np.concatenate([state, wrench_history(
                now[WRENCH], [s[pos[-k]][WRENCH] for k in self.history_lags])])
        out = {**data, "state": state.astype(np.float32)}
        if self.future_wrench and "actions" in data:
            # actions[t+k] == state[t+1+k] on the repaired data, so the wrench paired with
            # action step k is state offset 1+k (clamped at the episode end like the chunk).
            fut = np.stack([s[pos[k]][WRENCH] for k in range(1, self.action_horizon + 1)])
            out["actions"] = np.concatenate(
                [np.asarray(data["actions"])[:, :7], fut], axis=-1).astype(np.float32)
        return out


@dataclasses.dataclass(frozen=True)
class Forcevla_outputs(transforms.DataTransformFn):
    """
    This class is used to convert outputs from the model back the the dataset specific format. It is
    used for inference only.
    For your own dataset, you can copy this class and modify the action dimension based on the comments below.
    """
    # M8: also return the co-generated wrench (action dims 7:13, N) as `wrench_pred`.
    # Consumers index res["actions"], so the extra key is inert for them.
    emit_wrench: bool = False

    def __call__(self, data: dict) -> dict:
        # Only return the first N actions -- since we padded actions above to fit the model action
        # dimension, we need to now parse out the correct number of actions in the return dict.
        # For forcevla, we only return the first 7 actions (since the rest is padding), xyz  + RPY + gripper
        # For your own dataset, replace `7` with the action dimension of your dataset.
        out = {"actions": np.asarray(data["actions"][:, :7])}
        if self.emit_wrench:
            out["wrench_pred"] = np.asarray(data["actions"][:, 7:13])
        return out

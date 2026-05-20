#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_policy
from lerobot.utils.control_utils import predict_action


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate ACT policy offline inference with dummy observations.")
    parser.add_argument("--policy-path", required=True, help="Path to the pretrained policy directory.")
    parser.add_argument("--policy-device", default=None, choices=["cpu", "cuda", "mps"], help="Device to run inference on. Default uses policy config auto-selection.")
    parser.add_argument("--policy-use-amp", action="store_true", help="Enable automatic mixed precision during inference if the device supports it.")
    parser.add_argument("--dataset-root", default=None, help="Optional local dataset root containing stats for policy normalization.")
    parser.add_argument("--task", default="", help="Optional task string to use for action prediction.")
    parser.add_argument("--robot-type", default="", help="Optional robot type string to include in the observation.")
    return parser.parse_args()


def build_dummy_observation(config: PreTrainedConfig) -> dict[str, np.ndarray]:
    observation = {}

    # Visual inputs: convert channel-first shape to HWC for the dummy image.
    for key, ft in config.image_features.items():
        c, h, w = ft.shape
        observation[key] = np.zeros((h, w, c), dtype=np.uint8)

    # Non-visual inputs: use zeros in float32.
    for key, ft in config.input_features.items():
        if key in config.image_features:
            continue
        observation[key] = np.zeros(tuple(ft.shape), dtype=np.float32)

    return observation


def load_policy(policy_path: Path, device: str | None, use_amp: bool, dataset_root: str | None) -> ACTPolicy:
    cfg = ACTConfig.from_pretrained(policy_path)
    cfg.pretrained_path = str(policy_path)
    if device is not None:
        cfg.device = device
    cfg.use_amp = use_amp

    if dataset_root is not None:
        dataset_meta = LeRobotDatasetMetadata(cfg.repo_id or "unknown", root=dataset_root)
        return make_policy(cfg, ds_meta=dataset_meta)

    return ACTPolicy.from_pretrained(str(policy_path), config=cfg, strict=False)


def main() -> None:
    args = parse_args()
    policy_path = Path(args.policy_path)

    policy = load_policy(policy_path, args.policy_device, args.policy_use_amp, args.dataset_root)
    observation = build_dummy_observation(policy.config)

    print("Policy loaded:", type(policy).__name__)
    print("Device:", policy.config.device)
    print("Inputs:", {k: v.shape for k, v in observation.items()})
    print("Image features:", list(policy.config.image_features.keys()))

    try:
        action = predict_action(
            observation,
            policy,
            torch.device(policy.config.device),
            policy.config.use_amp,
            task=args.task,
            robot_type=args.robot_type if args.robot_type else None,
        )
        print("Action shape:", tuple(action.shape))
        print("Action values:", action.numpy())
    except AssertionError as e:
        print("AssertionError during inference:", e)
        print(
            "This usually means policy normalization stats are missing or not initialized."
            " Provide a dataset with stats or use a policy file that contains normalization buffers."
        )
    except Exception as e:
        print("Inference failed:", type(e).__name__, e)


if __name__ == "__main__":
    main()

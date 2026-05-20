#!/usr/bin/env python3
# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
import os
import sys
from pathlib import Path

# Jetson optimization: limit thread creation before loading torch.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.policies.factory import get_policy_class
from lerobot.utils.control_utils import predict_action
from lerobot.utils.utils import get_safe_torch_device
from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
from lerobot.robots.so101_follower.so101_follower import SO101Follower


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a pretrained ACT policy on a SO101 follower robot with CUDA on Jetson."
    )
    parser.add_argument("--policy-path", required=True, help="Path to the pretrained policy directory.")
    parser.add_argument("--robot-port", required=True, help="Serial port for the SO101 follower robot.")
    parser.add_argument("--robot-id", default="my_follower", help="Unique robot id used for calibration storage.")
    parser.add_argument("--camera-name", default=None, help="Camera key expected by the policy config.")
    parser.add_argument("--camera-index", type=int, default=0, help="OpenCV camera index.")
    parser.add_argument("--camera-width", type=int, default=640, help="Camera frame width.")
    parser.add_argument("--camera-height", type=int, default=480, help="Camera frame height.")
    parser.add_argument("--camera-fps", type=int, default=15, help="Requested camera FPS.")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu", "mps"], help="Torch device to use.")
    parser.add_argument("--use-amp", action="store_true", help="Enable automatic mixed precision for CUDA inference.")
    parser.add_argument("--task", default="", help="Task string to pass to the policy during inference.")
    parser.add_argument(
        "--max-relative-target",
        type=float,
        default=None,
        help="Max relative target for safety when sending joint goals.",
    )
    parser.add_argument(
        "--use-degrees",
        action="store_true",
        help="Use degrees for motor position normalization if the dataset/policy was trained that way.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=0,
        help="Number of control steps to run. Use 0 to run until interrupted.",
    )
    return parser.parse_args()


def build_observation_frame(observation: dict[str, np.ndarray], robot: SO101Follower, policy) -> dict[str, np.ndarray]:
    frame: dict[str, np.ndarray] = {}

    if "observation.state" in policy.config.input_features:
        state_keys = list(robot.action_features)
        frame["observation.state"] = np.array([observation[key] for key in state_keys], dtype=np.float32)

    for key in policy.config.image_features:
        assert key.startswith("observation.images."), f"Unsupported image feature key: {key}"
        camera_name = key.removeprefix("observation.images.")
        if camera_name not in observation:
            raise KeyError(
                f"Policy expects camera '{camera_name}' but robot observation contained: {list(observation.keys())}"
            )
        frame[key] = observation[camera_name]

    return frame


def build_action_dict(action_tensor: torch.Tensor, robot: SO101Follower) -> dict[str, float]:
    action_values = action_tensor.cpu().numpy().astype(np.float32).reshape(-1)
    action_keys = list(robot.action_features)
    if len(action_values) != len(action_keys):
        raise ValueError(
            f"Expected action dimension {len(action_keys)}, but policy output has shape {action_values.shape}."
        )
    return {key: float(value) for key, value in zip(action_keys, action_values)}


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        logging.warning("CUDA is not available. Falling back to CPU.")
        args.device = "cpu"

    if args.device == "mps" and not torch.backends.mps.is_available():
        logging.warning("MPS is not available. Falling back to CPU.")
        args.device = "cpu"

    device = get_safe_torch_device(args.device, log=True)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cudnn.benchmark = True

    policy_path = Path(args.policy_path)
    if not policy_path.exists():
        raise FileNotFoundError(f"Policy path not found: {policy_path}")

    policy_cfg = PreTrainedConfig.from_pretrained(policy_path)
    policy_cfg.device = args.device
    policy_cfg.use_amp = args.use_amp

    policy_cls = get_policy_class(policy_cfg.type)
    policy = policy_cls.from_pretrained(policy_path, config=policy_cfg)
    policy.reset()

    logging.info("Loaded pretrained policy %s", policy_cfg.type)
    logging.info("Policy device: %s", policy_cfg.device)
    logging.info("Policy input keys: %s", sorted(policy_cfg.input_features.keys()))
    logging.info("Policy output keys: %s", sorted(policy_cfg.output_features.keys()))

    image_features = list(policy_cfg.image_features.keys())
    if args.camera_name is None:
        if len(image_features) == 1:
            args.camera_name = image_features[0].removeprefix("observation.images.")
            logging.info("Inferred camera name '%s' from policy config.", args.camera_name)
        elif len(image_features) == 0:
            logging.info("Policy does not require image input.")
        else:
            raise ValueError(
                "Policy expects multiple image inputs. Please provide --camera-name to match policy image feature keys."
            )

    cameras = {}
    if args.camera_name is not None:
        cameras[args.camera_name] = OpenCVCameraConfig(
            args.camera_index,
            args.camera_fps,
            args.camera_width,
            args.camera_height,
        )

    robot_cfg = SO101FollowerConfig(
        port=args.robot_port,
        id=args.robot_id,
        cameras=cameras,
        disable_torque_on_disconnect=True,
        max_relative_target=args.max_relative_target,
        use_degrees=args.use_degrees,
    )
    robot = SO101Follower(robot_cfg)

    logging.info("Connecting robot on port %s", args.robot_port)
    robot.connect()
    logging.info("Robot connected. Starting inference loop.")

    try:
        step = 0
        while args.steps == 0 or step < args.steps:
            observation = robot.get_observation()
            frame = build_observation_frame(observation, robot, policy)
            action_tensor = predict_action(
                frame,
                policy,
                device,
                args.use_amp,
                task=args.task,
                robot_type=robot.robot_type,
            )
            action_dict = build_action_dict(action_tensor, robot)
            sent_action = robot.send_action(action_dict)
            logging.info("Step %s action=%s", step, sent_action)
            step += 1
    except KeyboardInterrupt:
        logging.info("Interrupted by user.")
    finally:
        logging.info("Disconnecting robot.")
        robot.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

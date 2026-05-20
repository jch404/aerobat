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
from pathlib import Path

# Jetson optimization: limit thread creation before loading torch.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import torch

try:
    import cv2
except ImportError:
    cv2 = None

from lerobot.configs.policies import PreTrainedConfig
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.policies.factory import get_policy_class
from lerobot.utils.control_utils import predict_action
from lerobot.utils.utils import get_safe_torch_device
from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
from lerobot.robots.so101_follower.so101_follower import SO101Follower

DEFAULT_POLICY_PATH = Path(
    "/home/user/project/lerobot/outputs/train/act_floor1_stack/checkpoints/080000/pretrained_model"
)
DEFAULT_ROBOT_PORT = "/dev/ttyACM0"
DEFAULT_ROBOT_ID = "my_follower"
DEFAULT_CAMERA_WIDTH = 640
DEFAULT_CAMERA_HEIGHT = 480
DEFAULT_CAMERA_FPS = 15
DEFAULT_DEVICE = "cuda"
DEFAULT_TASK = "test"
DEFAULT_CAMERA_DETECT_MAX_INDEX = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a pretrained ACT policy on a SO101 follower robot with CUDA on Jetson."
    )
    parser.add_argument(
        "--policy-path",
        default=str(DEFAULT_POLICY_PATH),
        help="Path to the pretrained policy directory.",
    )
    parser.add_argument(
        "--robot-port",
        default=DEFAULT_ROBOT_PORT,
        help="Serial port for the SO101 follower robot.",
    )
    parser.add_argument("--robot-id", default=DEFAULT_ROBOT_ID, help="Unique robot id used for calibration storage.")
    parser.add_argument(
        "--camera-config",
        action="append",
        default=[],
        help=(
            "Camera mapping in the form front=0 or top=1. "
            "Repeat for multiple cameras."
        ),
    )
    parser.add_argument(
        "--camera-name",
        action="append",
        default=[],
        help=(
            "Legacy camera name(s) for a single mapping. "
            "Repeat with --camera-index for multiple cameras."
        ),
    )
    parser.add_argument(
        "--camera-index",
        action="append",
        type=int,
        default=[],
        help="Legacy camera index(s) paired with --camera-name.",
    )
    parser.add_argument("--camera-width", type=int, default=DEFAULT_CAMERA_WIDTH, help="Camera frame width.")
    parser.add_argument("--camera-height", type=int, default=DEFAULT_CAMERA_HEIGHT, help="Camera frame height.")
    parser.add_argument("--camera-fps", type=int, default=DEFAULT_CAMERA_FPS, help="Requested camera FPS.")
    parser.add_argument("--device", default=DEFAULT_DEVICE, choices=["cuda", "cpu", "mps"], help="Torch device to use.")
    parser.add_argument(
        "--use-amp",
        action="store_true",
        default=True,
        help="Enable automatic mixed precision for CUDA inference.",
    )
    parser.add_argument("--no-amp", dest="use_amp", action="store_false", help="Disable automatic mixed precision.")
    parser.add_argument("--task", default=DEFAULT_TASK, help="Task string to pass to the policy during inference.")
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
    parser.add_argument(
        "--detect-cameras",
        action="store_true",
        help="Probe OpenCV camera indices 0..5 and print available cameras.",
    )
    return parser.parse_args()


def detect_connected_cameras(max_index: int = DEFAULT_CAMERA_DETECT_MAX_INDEX) -> list[int]:
    if cv2 is None:
        logging.warning("OpenCV is not installed. Camera detection is disabled.")
        return []

    available = []
    for index in range(max_index + 1):
        capture = cv2.VideoCapture(index)
        if capture is None:
            continue
        opened = capture.isOpened()
        capture.release()
        if opened:
            available.append(index)
    return available


def validate_paths(policy_path: Path, robot_port: str) -> None:
    if not policy_path.exists():
        raise FileNotFoundError(f"Policy path not found: {policy_path}")
    if not policy_path.is_dir():
        raise FileNotFoundError(f"Policy path must be a directory: {policy_path}")

    config_file = policy_path / "config.json"
    safetensor_file = policy_path / "model.safetensors"
    if not config_file.exists():
        raise FileNotFoundError(f"Policy config.json not found in {policy_path}")
    if not safetensor_file.exists():
        raise FileNotFoundError(
            f"Policy weights not found in {policy_path}. Expected {safetensor_file.name}."
        )

    if not Path(robot_port).exists():
        raise FileNotFoundError(f"Robot port not found: {robot_port}")


def parse_camera_mapping(args: argparse.Namespace) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for config_str in args.camera_config:
        if "=" not in config_str:
            raise ValueError(
                "Invalid --camera-config format. Use front=0 or top=1."
            )
        name, index_str = config_str.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError("Camera name cannot be empty in --camera-config.")
        try:
            mapping[name] = int(index_str)
        except ValueError as exc:
            raise ValueError(f"Camera index must be an integer in --camera-config: {config_str}") from exc

    if args.camera_name or args.camera_index:
        if len(args.camera_name) != len(args.camera_index):
            raise ValueError(
                "When using --camera-name and --camera-index together, the counts must match."
            )
        for name, index in zip(args.camera_name, args.camera_index):
            mapping[name] = index
    return mapping


def create_robot_camera_configs(required_image_features: list[str], camera_mapping: dict[str, int], args: argparse.Namespace) -> dict[str, OpenCVCameraConfig]:
    cameras: dict[str, OpenCVCameraConfig] = {}
    required_names = [key.removeprefix("observation.images.") for key in required_image_features]

    if not required_names:
        return cameras

    if not camera_mapping and len(required_names) > 1:
        available = detect_connected_cameras()
        raise ValueError(
            "The policy requires multiple image inputs (" + ", ".join(required_names) + "). "
            "Provide --camera-config front=0 --camera-config top=1 or use --camera-name/--camera-index. "
            f"Detected camera indices: {available if available else 'none'}."
        )

    if camera_mapping:
        missing = [name for name in required_names if name not in camera_mapping]
        if missing:
            available = detect_connected_cameras()
            raise ValueError(
                "Missing camera mapping for features: "
                + ", ".join(f"observation.images.{name}" for name in missing)
                + ". Provide --camera-config for each required camera. "
                f"Detected camera indices: {available if available else 'none'}."
            )

        for name in required_names:
            cameras[name] = OpenCVCameraConfig(
                camera_mapping[name],
                args.camera_fps,
                args.camera_width,
                args.camera_height,
            )
        return cameras

    if len(required_names) == 1:
        cameras[required_names[0]] = OpenCVCameraConfig(
            args.camera_index[0] if args.camera_index else 0,
            args.camera_fps,
            args.camera_width,
            args.camera_height,
        )
        return cameras

    raise ValueError(
        "Unable to build camera configuration. Please provide --camera-config for multiple image inputs."
    )


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
                "Missing camera observation for feature '" + key + "'. "
                f"Available observation keys: {list(observation.keys())}."
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

    logging.info("Jetson optimization: OMP_NUM_THREADS=%s OPENBLAS_NUM_THREADS=%s MKL_NUM_THREADS=%s NUMEXPR_NUM_THREADS=%s",
                 os.environ.get("OMP_NUM_THREADS"),
                 os.environ.get("OPENBLAS_NUM_THREADS"),
                 os.environ.get("MKL_NUM_THREADS"),
                 os.environ.get("NUMEXPR_NUM_THREADS"),
    )

    logging.info("Running with configuration:")
    logging.info("  policy_path=%s", args.policy_path)
    logging.info("  robot_port=%s", args.robot_port)
    logging.info("  robot_id=%s", args.robot_id)
    logging.info("  camera_config=%s", args.camera_config)
    logging.info("  camera_name=%s", args.camera_name)
    logging.info("  camera_index=%s", args.camera_index)
    logging.info("  camera_width=%s", args.camera_width)
    logging.info("  camera_height=%s", args.camera_height)
    logging.info("  camera_fps=%s", args.camera_fps)
    logging.info("  device=%s", args.device)
    logging.info("  use_amp=%s", args.use_amp)
    logging.info("  task=%s", args.task)
    logging.info("  steps=%s", args.steps)
    logging.info("  detect_cameras=%s", args.detect_cameras)

    if args.detect_cameras:
        available = detect_connected_cameras()
        logging.info("Detected camera indices: %s", available)

    validate_paths(Path(args.policy_path), args.robot_port)

    if args.device == "cuda" and not torch.cuda.is_available():
        logging.warning("CUDA is not available. Falling back to CPU.")
        args.device = "cpu"
    if args.device == "mps" and not torch.backends.mps.is_available():
        logging.warning("MPS is not available. Falling back to CPU.")
        args.device = "cpu"

    logging.info("CUDA available: %s", torch.cuda.is_available())
    logging.info("Requested device: %s", args.device)

    device = get_safe_torch_device(args.device, log=True)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cudnn.benchmark = True

    policy_path = Path(args.policy_path)
    policy_cfg = PreTrainedConfig.from_pretrained(policy_path)
    policy_cfg.device = args.device
    policy_cfg.use_amp = args.use_amp

    if args.use_amp and args.device != "cuda":
        logging.warning("Automatic mixed precision requested but only available on CUDA. Disabling AMP.")
        policy_cfg.use_amp = False

    policy_cls = get_policy_class(policy_cfg.type)
    policy = policy_cls.from_pretrained(policy_path, config=policy_cfg)
    policy.reset()

    logging.info("Loaded pretrained policy '%s'", policy_cfg.type)
    logging.info("Policy device: %s", policy_cfg.device)
    logging.info("Policy input features: %s", sorted(policy_cfg.input_features.keys()))
    logging.info("Policy output features: %s", sorted(policy_cfg.output_features.keys()))

    image_features = sorted(policy_cfg.image_features.keys())
    camera_mapping = parse_camera_mapping(args)
    cameras = create_robot_camera_configs(image_features, camera_mapping, args)
    logging.info("Using camera mappings: %s", {name: cfg.index_or_path for name, cfg in cameras.items()})

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
    try:
        robot.connect()
    except Exception:
        logging.exception("Robot connection failed")
        raise

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
                policy_cfg.use_amp,
                task=args.task,
                robot_type=robot.robot_type,
            )
            action_dict = build_action_dict(action_tensor, robot)
            sent_action = robot.send_action(action_dict)
            logging.info("Step %s action=%s", step, sent_action)
            step += 1
    except KeyboardInterrupt:
        logging.info("Interrupted by user.")
    except Exception:
        logging.exception("Inference loop failed")
        raise
    finally:
        logging.info("Disconnecting robot.")
        try:
            robot.disconnect()
        except Exception:
            logging.exception("Failed to disconnect robot cleanly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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
import gc
import json
import logging
import os
import socket
import time
from enum import Enum
from pathlib import Path

# Jetson optimization: limit thread creation before loading torch.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

import numpy as np
import torch
from safetensors.torch import load_file as load_safetensors_file

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.compute_stats import aggregate_stats
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.utils import (
    EPISODES_STATS_PATH,
    STATS_PATH,
    load_episodes_stats,
    load_info,
    load_stats,
    unflatten_dict,
)
from lerobot.policies.factory import get_policy_class
from lerobot.utils.control_utils import predict_action
from lerobot.utils.utils import get_safe_torch_device
from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
from lerobot.robots.so101_follower.so101_follower import SO101Follower

try:
    import cv2
except ImportError:
    cv2 = None

DEFAULT_POLICY_PATH = Path(
    "/home/user/project/lerobot/outputs/train/act_floor1_stack/checkpoints/080000/pretrained_model"
)
DEFAULT_ROBOT_PORT = "/dev/ttyACM0"
DEFAULT_ROBOT_ID = "my_follower"
DEFAULT_CAMERA_WIDTH = 640
DEFAULT_CAMERA_HEIGHT = 480
DEFAULT_CAMERA_FPS = 30
DEFAULT_DEVICE = "cuda"
DEFAULT_TASK = "test"
DEFAULT_CAMERA_DETECT_MAX_INDEX = 5
DEFAULT_TRIGGER_HOST = "0.0.0.0"
DEFAULT_TRIGGER_PORT = 8765
DEFAULT_PLACEMENT_TASKS = [
    "stack blue box on floor 1",
    "stack blue box on floor 2",
    "stack blue box beside floor 2 on floor 1",
]


class RobotRunState(Enum):
    WAIT = "WAIT"
    RUNNING = "RUNNING"
    RETURN_HOME = "RETURN_HOME"
    DONE = "DONE"


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
        "--action-scale",
        type=float,
        default=1.0,
        help="Scale policy actions before sending them to the robot. Use values like 0.3 for cautious tests.",
    )
    parser.add_argument(
        "--invert-action-all",
        action="store_true",
        help="Invert the sign of every action dimension before sending it to the robot.",
    )
    parser.add_argument(
        "--invert-action",
        action="append",
        default=[],
        help=(
            "Invert selected action keys. Accepts motor names with or without .pos, "
            "and comma-separated values. Example: --invert-action shoulder_pan,elbow_flex"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run policy inference and log actions without sending commands to the robot.",
    )
    parser.add_argument(
        "--debug-first-step",
        action="store_true",
        help="Log detailed observation/state/action mapping information on the first inference step.",
    )
    parser.add_argument(
        "--flip-camera",
        action="append",
        default=[],
        help=(
            "Flip a camera image before inference. Format: front=horizontal, top=vertical, or front=both. "
            "Repeat for multiple cameras."
        ),
    )
    parser.add_argument(
        "--use-degrees",
        action="store_true",
        help="Use degrees for motor position normalization if the dataset/policy was trained that way.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=10,
        help=(
            "Number of ACT control steps for one triggered stacking run. "
            "In immediate mode, use 0 to load policy/cameras/robot, then exit without inference."
        ),
    )
    parser.add_argument(
        "--run-forever",
        action="store_true",
        help="Run immediate ACT inference until interrupted. This replaces the old --steps 0 forever behavior.",
    )
    parser.add_argument(
        "--mcp-trigger-tcp",
        action="store_true",
        help="Wait for MCP TCP 'run' signals instead of running ACT inference continuously.",
    )
    parser.add_argument(
        "--trigger-host",
        default=DEFAULT_TRIGGER_HOST,
        help="Host/IP to bind for MCP TCP trigger signals.",
    )
    parser.add_argument(
        "--trigger-port",
        type=int,
        default=DEFAULT_TRIGGER_PORT,
        help="TCP port to bind for MCP trigger signals.",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=3,
        help="Maximum number of triggered stacking runs before DONE.",
    )
    parser.add_argument(
        "--placement-task",
        action="append",
        default=[],
        help=(
            "Task string for each stacking run. Repeat up to --max-runs times. "
            "Defaults are floor 1, floor 2, and floor-1 beside floor 2."
        ),
    )
    parser.add_argument(
        "--return-home-steps",
        type=int,
        default=30,
        help="Number of repeated home-position commands after each triggered run.",
    )
    parser.add_argument(
        "--control-period-s",
        type=float,
        default=0.0,
        help="Optional sleep time between robot commands.",
    )
    parser.add_argument(
        "--wait-reset-after-done",
        action="store_true",
        help="After --max-runs, keep the TCP server alive and wait for a 'reset' signal.",
    )
    parser.add_argument(
        "--detect-cameras",
        action="store_true",
        help="Probe OpenCV camera indices 0..5 and print available cameras.",
    )
    parser.add_argument(
        "--dataset-repo-id",
        default=None,
        help=(
            "Optional dataset repo id to load normalization stats from. "
            "If omitted, the runner will try to infer the dataset from train_config.json."
        ),
    )
    parser.add_argument(
        "--dataset-root",
        default=None,
        help=(
            "Optional local dataset root path to load dataset metadata and normalization stats. "
            "If omitted, the dataset root from train_config.json will be used when available."
        ),
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


def log_memory_status(label: str) -> None:
    """Log lightweight process/GPU memory information when available."""
    try:
        import resource

        peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        logging.info("Memory %s: process_peak_rss=%.1f MiB", label, peak_mb)
    except Exception:
        logging.debug("Unable to read process memory status", exc_info=True)

    if torch.cuda.is_available():
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            logging.info(
                "CUDA memory %s: free=%.1f MiB total=%.1f MiB allocated=%.1f MiB reserved=%.1f MiB",
                label,
                free_bytes / 1024 / 1024,
                total_bytes / 1024 / 1024,
                torch.cuda.memory_allocated() / 1024 / 1024,
                torch.cuda.memory_reserved() / 1024 / 1024,
            )
        except Exception:
            logging.debug("Unable to read CUDA memory status", exc_info=True)


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


def find_train_config_path(policy_path: Path) -> Path | None:
    for candidate in [
        policy_path / "train_config.json",
        policy_path.parent / "train_config.json",
        policy_path.parent.parent / "train_config.json",
    ]:
        if candidate.exists():
            return candidate
    return None


def resolve_dataset_root(root_value: str | None, base_path: Path) -> str | None:
    if root_value is None:
        return None
    root_path = Path(root_value)
    if root_path.is_absolute():
        return str(root_path)
    return str((base_path.parent / root_path).resolve())


def extract_dataset_info_from_train_config(train_config_path: Path) -> tuple[str | None, str | None]:
    try:
        with open(train_config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    except Exception as exc:
        logging.warning("Unable to read train_config.json at %s: %s", train_config_path, exc)
        return None, None

    if not isinstance(config, dict):
        return None, None

    dataset_config = config.get("dataset")
    if not isinstance(dataset_config, dict):
        return None, None

    repo_id = dataset_config.get("repo_id")
    if isinstance(repo_id, list):
        repo_id = repo_id[0] if repo_id else None

    root = dataset_config.get("root")
    root = resolve_dataset_root(root, train_config_path) if isinstance(root, str) else None
    return repo_id, root


def load_dataset_stats_from_root(dataset_root: str | None) -> dict | None:
    if dataset_root is None:
        return None

    root_path = Path(dataset_root)
    if not root_path.exists():
        logging.warning("Dataset root path does not exist: %s", root_path)
        return None

    stats = load_stats(root_path)
    if stats is not None:
        logging.info("Loaded normalization stats from %s", root_path / STATS_PATH)
        return stats

    try:
        episodes_stats = load_episodes_stats(root_path)
        stats = aggregate_stats(list(episodes_stats.values()))
        logging.info("Loaded normalization stats by aggregating %s", root_path / EPISODES_STATS_PATH)
        return stats
    except Exception as exc:
        logging.warning("Failed to load aggregate stats from dataset root %s: %s", root_path, exc)
        return None


def load_stats_file(stats_path: Path) -> dict | None:
    if not stats_path.exists():
        return None

    if stats_path.suffix == ".json":
        try:
            with open(stats_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except Exception as exc:
            logging.warning("Failed to load JSON stats from %s: %s", stats_path, exc)
        return None

    if stats_path.suffix == ".safetensors":
        try:
            tensors = load_safetensors_file(str(stats_path))
            return unflatten_dict({key: tensor for key, tensor in tensors.items()})
        except Exception as exc:
            logging.warning("Failed to load safetensors stats from %s: %s", stats_path, exc)
        return None

    if stats_path.suffix == ".npz":
        try:
            data = np.load(stats_path, allow_pickle=True)
            return {key: data[key] for key in data.files}
        except Exception as exc:
            logging.warning("Failed to load NPZ stats from %s: %s", stats_path, exc)
        return None

    return None


def find_stats_in_directory(directory: Path) -> Path | None:
    candidates = [
        directory / "meta" / "stats.json",
        directory / "meta" / "episodes_stats.jsonl",
        directory / "meta_data" / "stats.safetensors",
        directory / "stats.safetensors",
        directory / "stats.json",
        directory / "meta" / "stats.safetensors",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def auto_find_dataset_root(policy_path: Path) -> str | None:
    for candidate_root in [policy_path, policy_path.parent, policy_path.parent.parent, Path.cwd()]:
        stats_file = find_stats_in_directory(candidate_root)
        if stats_file is not None:
            return str(candidate_root)
    return None


def create_identity_stats(config: PreTrainedConfig, stats: dict | None = None) -> dict:
    stats = {} if stats is None else dict(stats)
    for feature_map in (config.input_features, config.output_features):
        for key, ft in feature_map.items():
            if key in stats:
                continue
            norm_mode = config.normalization_mapping.get(ft.type, NormalizationMode.IDENTITY)
            if norm_mode is NormalizationMode.IDENTITY:
                continue

            shape = tuple(ft.shape)
            if ft.type is FeatureType.VISUAL:
                shape = (shape[0], 1, 1)

            if norm_mode is NormalizationMode.MEAN_STD:
                stats[key] = {
                    "mean": np.zeros(shape, dtype=np.float32),
                    "std": np.ones(shape, dtype=np.float32),
                }
            elif norm_mode is NormalizationMode.MIN_MAX:
                stats[key] = {
                    "min": np.zeros(shape, dtype=np.float32),
                    "max": np.ones(shape, dtype=np.float32),
                }
    return stats



def repair_missing_normalization_buffers(policy) -> None:
    """Replace inf/nan normalization buffers with safe identity values.

    This is a Jetson/inference safety fallback. Some old LeRobot checkpoints do
    not include dataset normalization buffers in model.safetensors, and newer
    LeRobot versions initialize those missing buffers with inf. ACT then crashes
    at normalize.py with: AssertionError: `mean` is infinity.

    Safe fallback:
    - mean/min -> 0
    - std/max  -> 1

    This may reduce policy quality if the real dataset stats are missing, but it
    allows inference to run and confirms the hardware/action path.
    """
    repaired = []

    for module_name, module in policy.named_modules():
        for buffer_name, buffer in module.named_buffers(recurse=False):
            if not torch.is_tensor(buffer):
                continue

            bad = torch.isnan(buffer).any() or torch.isinf(buffer).any()
            if not bad:
                continue

            lowered = buffer_name.lower()
            if "std" in lowered or "max" in lowered:
                fill_value = 1.0
            else:
                fill_value = 0.0

            with torch.no_grad():
                buffer.data = torch.full_like(buffer, fill_value)

            full_name = f"{module_name}.{buffer_name}" if module_name else buffer_name
            repaired.append(f"{full_name}={fill_value}")

    if repaired:
        logging.warning(
            "Repaired missing/invalid normalization buffers with identity fallback: %s",
            ", ".join(repaired),
        )
    else:
        logging.info("Normalization buffers look valid; no repair needed.")


def print_policy_normalization_status(policy) -> None:
    bad_buffers = []
    for module_name, module in policy.named_modules():
        for buffer_name, buffer in module.named_buffers(recurse=False):
            if not torch.is_tensor(buffer):
                continue
            if torch.isnan(buffer).any() or torch.isinf(buffer).any():
                full_name = f"{module_name}.{buffer_name}" if module_name else buffer_name
                bad_buffers.append(full_name)

    if bad_buffers:
        logging.warning("Still invalid normalization buffers: %s", ", ".join(bad_buffers))
    else:
        logging.info("All normalization buffers are finite.")



def load_dataset_stats(policy_path: Path, dataset_repo_id: str | None, dataset_root: str | None) -> dict | None:
    """Load dataset normalization stats from train_config.json or explicit dataset metadata."""
    train_config_path = find_train_config_path(policy_path)
    if train_config_path is not None:
        logging.info("Found train_config.json at %s", train_config_path)
        repo_id, root = extract_dataset_info_from_train_config(train_config_path)
        if dataset_repo_id is None:
            dataset_repo_id = repo_id
        if dataset_root is None:
            dataset_root = root

    if dataset_root is not None:
        stats = load_dataset_stats_from_root(dataset_root)
        if stats is not None:
            return stats

    if dataset_repo_id is None:
        logging.warning(
            "No dataset repo id or root available for loading normalization stats. "
            "Use --dataset-repo-id or --dataset-root if train_config.json cannot be parsed."
        )
        return None

    try:
        ds_meta = LeRobotDatasetMetadata(repo_id=dataset_repo_id, root=dataset_root)
        logging.info(
            "Loaded dataset metadata from repo_id=%s root=%s revision=%s",
            ds_meta.repo_id,
            ds_meta.root,
            ds_meta.revision,
        )
        return ds_meta.stats
    except Exception as exc:
        logging.warning("Failed to load dataset metadata for normalization stats from repo_id=%s root=%s: %s", dataset_repo_id, dataset_root, exc)

    for candidate_root in [policy_path, policy_path.parent, policy_path.parent.parent]:
        stats_file = find_stats_in_directory(candidate_root)
        if stats_file is not None:
            stats = load_stats_file(stats_file)
            if stats is not None:
                logging.info("Loaded normalization stats from checkpoint directory %s", stats_file)
                return stats

    return None


def resolve_dataset_root_from_config(policy_path: Path, dataset_root: str | None) -> str | None:
    if dataset_root is not None:
        return dataset_root

    train_config_path = find_train_config_path(policy_path)
    if train_config_path is None:
        return None

    _, root = extract_dataset_info_from_train_config(train_config_path)
    return root


def normalize_motor_feature_name(name: str) -> str:
    return name if name.endswith(".pos") else f"{name}.pos"


def load_dataset_feature_order(dataset_root: str | None) -> tuple[list[str] | None, list[str] | None]:
    if dataset_root is None:
        return None, None

    root_path = Path(dataset_root)
    if not root_path.exists():
        logging.warning("Cannot load dataset feature order. Dataset root path does not exist: %s", root_path)
        return None, None

    try:
        info = load_info(root_path)
    except Exception as exc:
        logging.warning("Cannot load dataset feature order from %s/meta/info.json: %s", root_path, exc)
        return None, None

    features = info.get("features", {})
    state_names = features.get("observation.state", {}).get("names")
    action_names = features.get("action", {}).get("names")

    state_keys = [normalize_motor_feature_name(name) for name in state_names] if isinstance(state_names, list) else None
    action_keys = [normalize_motor_feature_name(name) for name in action_names] if isinstance(action_names, list) else None

    logging.info("Dataset observation.state feature order: %s", state_keys)
    logging.info("Dataset action feature order: %s", action_keys)
    return state_keys, action_keys


def validate_feature_order(feature_name: str, keys: list[str] | None, available_keys: list[str]) -> list[str] | None:
    if keys is None:
        return None

    missing = [key for key in keys if key not in available_keys]
    if missing:
        logging.warning(
            "Dataset %s feature order contains keys not available on this robot: missing=%s available=%s. "
            "Falling back to robot order.",
            feature_name,
            missing,
            available_keys,
        )
        return None

    if len(keys) != len(available_keys):
        logging.warning(
            "Dataset %s feature order length differs from robot keys: dataset=%s robot=%s. "
            "Falling back to robot order.",
            feature_name,
            keys,
            available_keys,
        )
        return None

    return keys


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


def parse_camera_flips(raw_values: list[str]) -> dict[str, str]:
    flips: dict[str, str] = {}
    valid_modes = {"horizontal", "vertical", "both"}
    for raw_value in raw_values:
        if "=" not in raw_value:
            raise ValueError("Invalid --flip-camera format. Use front=horizontal, top=vertical, or front=both.")
        name, mode = raw_value.split("=", 1)
        name = name.strip()
        mode = mode.strip().lower()
        if not name:
            raise ValueError("Camera name cannot be empty in --flip-camera.")
        if mode not in valid_modes:
            raise ValueError(
                f"Invalid flip mode '{mode}' for camera '{name}'. Valid modes: {sorted(valid_modes)}."
            )
        flips[name] = mode
    return flips


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
                index_or_path=camera_mapping[name],
                fps=args.camera_fps,
                width=args.camera_width,
                height=args.camera_height,
                color_mode="rgb",
            )
        return cameras

    if len(required_names) == 1:
        cameras[required_names[0]] = OpenCVCameraConfig(
            index_or_path=args.camera_index[0] if args.camera_index else 0,
            fps=args.camera_fps,
            width=args.camera_width,
            height=args.camera_height,
            color_mode="rgb",
        )
        return cameras

    raise ValueError(
        "Unable to build camera configuration. Please provide --camera-config for multiple image inputs."
    )


def get_state_keys(robot: SO101Follower, dataset_state_keys: list[str] | None = None) -> list[str]:
    return dataset_state_keys if dataset_state_keys is not None else list(robot.action_features)


def apply_camera_flip(image: np.ndarray, mode: str) -> np.ndarray:
    if mode == "horizontal":
        return np.flip(image, axis=1).copy()
    if mode == "vertical":
        return np.flip(image, axis=0).copy()
    if mode == "both":
        return np.flip(image, axis=(0, 1)).copy()
    raise ValueError(f"Unsupported camera flip mode: {mode}")


def build_observation_frame(
    observation: dict[str, np.ndarray],
    robot: SO101Follower,
    policy,
    camera_flips: dict[str, str] | None = None,
    dataset_state_keys: list[str] | None = None,
) -> dict[str, np.ndarray]:
    frame: dict[str, np.ndarray] = {}
    camera_flips = {} if camera_flips is None else camera_flips

    if "observation.state" in policy.config.input_features:
        state_keys = get_state_keys(robot, dataset_state_keys)
        frame["observation.state"] = np.array([observation[key] for key in state_keys], dtype=np.float32)

    for key in policy.config.image_features:
        assert key.startswith("observation.images."), f"Unsupported image feature key: {key}"
        camera_name = key.removeprefix("observation.images.")
        if camera_name not in observation:
            raise KeyError(
                "Missing camera observation for feature '" + key + "'. "
                f"Available observation keys: {list(observation.keys())}."
            )
        image = observation[camera_name]
        if camera_name in camera_flips:
            image = apply_camera_flip(image, camera_flips[camera_name])
        frame[key] = image

    return frame


def parse_action_keys(raw_values: list[str], action_keys: list[str]) -> set[str]:
    requested = []
    for raw_value in raw_values:
        requested.extend(part.strip() for part in raw_value.split(","))

    key_lookup = {key: key for key in action_keys}
    key_lookup.update({key.removesuffix(".pos"): key for key in action_keys})

    selected = set()
    unknown = []
    for name in requested:
        if not name:
            continue
        key = key_lookup.get(name)
        if key is None:
            unknown.append(name)
        else:
            selected.add(key)

    if unknown:
        raise ValueError(
            "Unknown --invert-action key(s): "
            + ", ".join(unknown)
            + ". Available action keys: "
            + ", ".join(action_keys)
        )
    return selected


def build_action_dict(
    action_tensor: torch.Tensor,
    robot: SO101Follower,
    dataset_action_keys: list[str] | None = None,
    action_scale: float = 1.0,
    invert_action_all: bool = False,
    invert_action_keys: set[str] | None = None,
) -> dict[str, float]:
    action_values = action_tensor.cpu().numpy().astype(np.float32).reshape(-1)
    action_keys = dataset_action_keys if dataset_action_keys is not None else list(robot.action_features)
    if len(action_values) != len(action_keys):
        raise ValueError(
            f"Expected action dimension {len(action_keys)}, but policy output has shape {action_values.shape}."
        )

    invert_action_keys = set() if invert_action_keys is None else invert_action_keys
    signs = np.array(
        [
            -1.0 if invert_action_all or key in invert_action_keys else 1.0
            for key in action_keys
        ],
        dtype=np.float32,
    )
    action_values = action_values * signs * action_scale
    return {key: float(value) for key, value in zip(action_keys, action_values)}


def tensor_to_float_list(tensor: torch.Tensor) -> list[float]:
    return tensor.detach().cpu().numpy().astype(np.float32).reshape(-1).tolist()


def log_policy_feature_config(policy_cfg: PreTrainedConfig) -> None:
    logging.info("Policy config input_features: %s", policy_cfg.input_features)
    logging.info("Policy config output_features: %s", policy_cfg.output_features)
    logging.info("Policy config image_features: %s", policy_cfg.image_features)


def log_robot_feature_config(robot: SO101Follower) -> None:
    logging.info("Robot action_features: %s", robot.action_features)
    logging.info("Robot observation_features: %s", robot.observation_features)


def log_first_step_debug(
    observation: dict[str, np.ndarray],
    frame: dict[str, np.ndarray],
    action_tensor: torch.Tensor,
    action_dict: dict[str, float],
    robot: SO101Follower,
    policy,
    dataset_state_keys: list[str] | None,
    dataset_action_keys: list[str] | None,
) -> None:
    state_keys = get_state_keys(robot, dataset_state_keys)
    logging.info("DEBUG first step observation.keys: %s", list(observation.keys()))
    logging.info("DEBUG first step state_keys used for observation.state: %s", state_keys)
    logging.info("DEBUG first step action_keys used for action_tensor mapping: %s", dataset_action_keys or list(robot.action_features))
    if "observation.state" in frame:
        logging.info("DEBUG first step observation.state values: %s", frame["observation.state"].tolist())
    for image_key in policy.config.image_features:
        if image_key in frame:
            image = frame[image_key]
            logging.info(
                "DEBUG first step %s shape=%s dtype=%s min=%s max=%s",
                image_key,
                getattr(image, "shape", None),
                getattr(image, "dtype", None),
                np.min(image) if isinstance(image, np.ndarray) else None,
                np.max(image) if isinstance(image, np.ndarray) else None,
            )
    logging.info("DEBUG first step raw action_tensor values: %s", tensor_to_float_list(action_tensor))
    logging.info("DEBUG first step action_dict sent mapping: %s", action_dict)


def log_state(state: RobotRunState, message: str, *args) -> None:
    logging.info("[%s] " + message, state.value, *args)


def resolve_placement_tasks(raw_tasks: list[str], max_runs: int) -> list[str]:
    tasks = raw_tasks if raw_tasks else DEFAULT_PLACEMENT_TASKS
    if len(tasks) < max_runs:
        raise ValueError(
            f"Need at least {max_runs} placement task(s), but got {len(tasks)}. "
            "Repeat --placement-task for each run."
        )
    return tasks[:max_runs]


def parse_trigger_command(payload: str) -> str | None:
    payload = payload.strip()
    if not payload:
        return None

    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        data = None

    if isinstance(data, dict):
        raw_command = data.get("command") or data.get("signal") or data.get("event")
        command = str(raw_command).strip().lower() if raw_command is not None else ""
    else:
        command = payload.splitlines()[0].strip().lower()

    if command in {"run", "start", "stack", "trigger"}:
        return "run"
    if command in {"reset", "restart"}:
        return "reset"
    return None


def read_trigger_command(client: socket.socket) -> str | None:
    client.settimeout(2.0)
    chunks = []
    while True:
        try:
            chunk = client.recv(4096)
        except socket.timeout:
            break
        if not chunk:
            break
        chunks.append(chunk)
        if b"\n" in chunk:
            break

    payload = b"".join(chunks).decode("utf-8", errors="replace")
    return parse_trigger_command(payload)


def send_trigger_response(client: socket.socket, message: str) -> None:
    try:
        client.sendall((message + "\n").encode("utf-8"))
    except OSError:
        logging.debug("Failed to send trigger response to MCP client", exc_info=True)


def create_trigger_server(host: str, port: int) -> socket.socket:
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((host, port))
    server.listen()
    server.settimeout(0.5)
    logging.info("MCP TCP trigger server listening on %s:%s", host, port)
    return server


def accept_trigger(server: socket.socket) -> str | None:
    try:
        client, address = server.accept()
    except socket.timeout:
        return None

    with client:
        command = read_trigger_command(client)
        if command is None:
            logging.warning("Ignored unknown MCP trigger from %s", address)
            send_trigger_response(client, "ignored unknown command")
            return None

        logging.info("Received MCP trigger '%s' from %s", command, address)
        send_trigger_response(client, f"accepted {command}")
        return command


def drain_pending_triggers(server: socket.socket, state: RobotRunState) -> None:
    previous_timeout = server.gettimeout()
    server.settimeout(0.0)
    try:
        while True:
            try:
                client, address = server.accept()
            except (BlockingIOError, socket.timeout):
                break
            with client:
                command = read_trigger_command(client)
                log_state(state, "Ignoring MCP trigger while busy: command=%s address=%s", command, address)
                send_trigger_response(client, f"ignored while {state.value}")
    finally:
        server.settimeout(previous_timeout)


def build_home_action(observation: dict[str, np.ndarray], action_keys: list[str]) -> dict[str, float]:
    missing = [key for key in action_keys if key not in observation]
    if missing:
        raise KeyError(f"Cannot build home action. Missing observation keys: {missing}")
    return {key: float(observation[key]) for key in action_keys}


def sleep_control_period(control_period_s: float) -> None:
    if control_period_s > 0:
        time.sleep(control_period_s)


def run_act_once(
    run_index: int,
    task: str,
    steps: int,
    robot: SO101Follower,
    policy,
    device: torch.device,
    use_amp: bool,
    action_scale: float,
    invert_action_all: bool,
    invert_action_keys: set[str],
    control_period_s: float,
    camera_flips: dict[str, str],
    dry_run: bool,
    dataset_state_keys: list[str] | None,
    dataset_action_keys: list[str] | None,
) -> None:
    if steps <= 0:
        raise ValueError("--steps must be greater than 0 in MCP trigger mode.")

    policy.reset()
    log_state(RobotRunState.RUNNING, "Run %s started. task=%s steps=%s", run_index, task, steps)
    for step in range(steps):
        observation = robot.get_observation()
        frame = build_observation_frame(
            observation,
            robot,
            policy,
            camera_flips=camera_flips,
            dataset_state_keys=dataset_state_keys,
        )
        with torch.inference_mode():
            action_tensor = predict_action(
                frame,
                policy,
                device,
                use_amp,
                task=task,
                robot_type=robot.robot_type,
            )
        action_dict = build_action_dict(
            action_tensor,
            robot,
            dataset_action_keys=dataset_action_keys,
            action_scale=action_scale,
            invert_action_all=invert_action_all,
            invert_action_keys=invert_action_keys,
        )
        logging.info("Run %s step %s raw_action_tensor=%s", run_index, step, tensor_to_float_list(action_tensor))
        logging.info("Run %s step %s action_dict=%s", run_index, step, action_dict)
        if dry_run:
            logging.info("Run %s step %s dry_run=true; action not sent to robot.", run_index, step)
        else:
            sent_action = robot.send_action(action_dict)
            logging.info("Run %s step %s sent_action=%s", run_index, step, sent_action)
        sleep_control_period(control_period_s)
    log_state(RobotRunState.RUNNING, "Run %s completed.", run_index)


def return_home(
    run_index: int,
    robot: SO101Follower,
    home_action: dict[str, float],
    return_home_steps: int,
    control_period_s: float,
) -> None:
    if return_home_steps <= 0:
        log_state(RobotRunState.RETURN_HOME, "Run %s return home skipped.", run_index)
        return

    log_state(RobotRunState.RETURN_HOME, "Run %s returning home. steps=%s", run_index, return_home_steps)
    for step in range(return_home_steps):
        sent_action = robot.send_action(home_action)
        logging.info("Return home run %s step %s action=%s", run_index, step, sent_action)
        sleep_control_period(control_period_s)


def run_immediate_loop(
    args: argparse.Namespace,
    robot: SO101Follower,
    policy,
    device: torch.device,
    use_amp: bool,
    invert_action_keys: set[str],
    camera_flips: dict[str, str],
    dataset_state_keys: list[str] | None,
    dataset_action_keys: list[str] | None,
) -> None:
    if args.steps == 0:
        logging.info("--steps 0 requested. Load check completed; skipping inference loop.")
        return

    logging.info("Robot connected. Starting immediate inference loop.")
    step = 0
    while args.run_forever or step < args.steps:
        observation = robot.get_observation()
        frame = build_observation_frame(
            observation,
            robot,
            policy,
            camera_flips=camera_flips,
            dataset_state_keys=dataset_state_keys,
        )
        with torch.inference_mode():
            action_tensor = predict_action(
                frame,
                policy,
                device,
                use_amp,
                task=args.task,
                robot_type=robot.robot_type,
            )
        action_dict = build_action_dict(
            action_tensor,
            robot,
            dataset_action_keys=dataset_action_keys,
            action_scale=args.action_scale,
            invert_action_all=args.invert_action_all,
            invert_action_keys=invert_action_keys,
        )
        if args.debug_first_step and step == 0:
            log_first_step_debug(
                observation,
                frame,
                action_tensor,
                action_dict,
                robot,
                policy,
                dataset_state_keys,
                dataset_action_keys,
            )
        logging.info("Step %s raw_action_tensor=%s", step, tensor_to_float_list(action_tensor))
        logging.info("Step %s action_dict=%s", step, action_dict)
        if args.dry_run:
            logging.info("Step %s dry_run=true; action not sent to robot.", step)
        else:
            sent_action = robot.send_action(action_dict)
            logging.info("Step %s sent_action=%s", step, sent_action)
        step += 1
        sleep_control_period(args.control_period_s)


def run_mcp_trigger_loop(
    args: argparse.Namespace,
    robot: SO101Follower,
    policy,
    device: torch.device,
    use_amp: bool,
    invert_action_keys: set[str],
    action_keys: list[str],
    camera_flips: dict[str, str],
    dataset_state_keys: list[str] | None,
    dataset_action_keys: list[str] | None,
) -> None:
    placement_tasks = resolve_placement_tasks(args.placement_task, args.max_runs)
    home_observation = robot.get_observation()
    home_action = build_home_action(home_observation, action_keys)

    completed_runs = 0
    last_wait_run = None
    done_logged = False
    with create_trigger_server(args.trigger_host, args.trigger_port) as server:
        while True:
            if completed_runs >= args.max_runs:
                if not done_logged:
                    log_state(RobotRunState.DONE, "Completed %s/%s runs.", completed_runs, args.max_runs)
                    done_logged = True
                if not args.wait_reset_after_done:
                    return

                command = accept_trigger(server)
                if command == "reset":
                    completed_runs = 0
                    last_wait_run = None
                    done_logged = False
                    log_state(RobotRunState.WAIT, "Reset received. Waiting for run signal.")
                elif command == "run":
                    log_state(RobotRunState.DONE, "Ignoring run signal after DONE. Send reset first.")
                continue

            next_run = completed_runs + 1
            next_task = placement_tasks[completed_runs]
            if last_wait_run != next_run:
                log_state(
                    RobotRunState.WAIT,
                    "Waiting for MCP run signal. next_run=%s/%s task=%s",
                    next_run,
                    args.max_runs,
                    next_task,
                )
                last_wait_run = next_run

            command = accept_trigger(server)
            if command is None:
                continue
            if command == "reset":
                completed_runs = 0
                last_wait_run = None
                log_state(RobotRunState.WAIT, "Reset received. Counter already reset.")
                continue
            if command != "run":
                continue

            run_act_once(
                next_run,
                next_task,
                args.steps,
                robot,
                policy,
                device,
                use_amp,
                args.action_scale,
                args.invert_action_all,
                invert_action_keys,
                args.control_period_s,
                camera_flips,
                args.dry_run,
                dataset_state_keys,
                dataset_action_keys,
            )
            drain_pending_triggers(server, RobotRunState.RUNNING)
            return_home(next_run, robot, home_action, args.return_home_steps, args.control_period_s)
            drain_pending_triggers(server, RobotRunState.RETURN_HOME)
            completed_runs += 1
            last_wait_run = None


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
    logging.info("  action_scale=%s", args.action_scale)
    logging.info("  invert_action_all=%s", args.invert_action_all)
    logging.info("  invert_action=%s", args.invert_action)
    logging.info("  dry_run=%s", args.dry_run)
    logging.info("  debug_first_step=%s", args.debug_first_step)
    logging.info("  flip_camera=%s", args.flip_camera)
    logging.info("  steps=%s", args.steps)
    logging.info("  run_forever=%s", args.run_forever)
    logging.info("  mcp_trigger_tcp=%s", args.mcp_trigger_tcp)
    logging.info("  trigger_host=%s", args.trigger_host)
    logging.info("  trigger_port=%s", args.trigger_port)
    logging.info("  max_runs=%s", args.max_runs)
    logging.info("  placement_task=%s", args.placement_task)
    logging.info("  return_home_steps=%s", args.return_home_steps)
    logging.info("  control_period_s=%s", args.control_period_s)
    logging.info("  wait_reset_after_done=%s", args.wait_reset_after_done)
    logging.info("  detect_cameras=%s", args.detect_cameras)

    if args.action_scale <= 0:
        raise ValueError("--action-scale must be greater than 0.")
    if args.steps < 0:
        raise ValueError("--steps must be 0 or greater. Use --run-forever for continuous immediate inference.")
    if args.max_runs <= 0:
        raise ValueError("--max-runs must be greater than 0.")
    if args.return_home_steps < 0:
        raise ValueError("--return-home-steps must be 0 or greater.")
    if args.control_period_s < 0:
        raise ValueError("--control-period-s must be 0 or greater.")

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
    log_memory_status("before policy load")

    device = get_safe_torch_device(args.device, log=True)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cudnn.benchmark = False

    policy_path = Path(args.policy_path)
    policy_cfg = PreTrainedConfig.from_pretrained(policy_path)
    policy_cfg.device = args.device
    policy_cfg.use_amp = args.use_amp
    log_policy_feature_config(policy_cfg)

    if args.use_amp and args.device != "cuda":
        logging.warning("Automatic mixed precision requested but only available on CUDA. Disabling AMP.")
        policy_cfg.use_amp = False

    dataset_stats = load_dataset_stats(policy_path, args.dataset_repo_id, args.dataset_root)
    if dataset_stats is None:
        logging.warning(
            "Dataset normalization stats were not loaded. Falling back to identity stats for any missing normalization features."
        )
        dataset_stats = create_identity_stats(policy_cfg)
    else:
        dataset_stats = create_identity_stats(policy_cfg, dataset_stats)

    policy_cls = get_policy_class(policy_cfg.type)
    policy = policy_cls.from_pretrained(policy_path, config=policy_cfg, dataset_stats=dataset_stats)
    policy.eval()
    del dataset_stats
    gc.collect()
    if args.device == "cuda":
        torch.cuda.empty_cache()
    log_memory_status("after policy load")

    # Important fallback for old/incompatible checkpoints:
    # Even when dataset_stats is provided, some LeRobot versions keep missing
    # normalizer buffers as inf after loading model.safetensors. Repair them
    # before the first policy.select_action() call.
    repair_missing_normalization_buffers(policy)
    print_policy_normalization_status(policy)

    policy.reset()

    logging.info("Loaded pretrained policy '%s'", policy_cfg.type)
    logging.info("Policy device: %s", policy_cfg.device)
    logging.info("Policy input features: %s", sorted(policy_cfg.input_features.keys()))
    logging.info("Policy output features: %s", sorted(policy_cfg.output_features.keys()))

    image_features = sorted(policy_cfg.image_features.keys())
    camera_mapping = parse_camera_mapping(args)
    camera_flips = parse_camera_flips(args.flip_camera)
    cameras = create_robot_camera_configs(image_features, camera_mapping, args)
    unknown_flip_cameras = sorted(set(camera_flips) - set(cameras))
    if unknown_flip_cameras:
        raise ValueError(
            "Unknown --flip-camera name(s): "
            + ", ".join(unknown_flip_cameras)
            + ". Configured cameras: "
            + ", ".join(sorted(cameras))
        )
    logging.info("Using camera mappings: %s", {name: cfg.index_or_path for name, cfg in cameras.items()})
    logging.info("Using camera flips: %s", camera_flips)
    for name, cfg in cameras.items():
        logging.info(
            "Camera %s: index=%s width=%s height=%s fps=%s color_mode=%s",
            name,
            cfg.index_or_path,
            cfg.width,
            cfg.height,
            cfg.fps,
            cfg.color_mode,
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
    robot_action_keys = list(robot.action_features)
    dataset_root_for_features = resolve_dataset_root_from_config(policy_path, args.dataset_root)
    dataset_state_keys, dataset_action_keys = load_dataset_feature_order(dataset_root_for_features)
    dataset_state_keys = validate_feature_order(
        "observation.state",
        dataset_state_keys,
        robot_action_keys,
    )
    dataset_action_keys = validate_feature_order("action", dataset_action_keys, robot_action_keys)
    action_keys = dataset_action_keys if dataset_action_keys is not None else robot_action_keys
    if dataset_state_keys is not None and dataset_state_keys != robot_action_keys:
        logging.warning(
            "Using dataset observation.state order because it differs from robot order: dataset=%s robot=%s",
            dataset_state_keys,
            robot_action_keys,
        )
    if dataset_action_keys is not None and dataset_action_keys != robot_action_keys:
        logging.warning(
            "Using dataset action order because it differs from robot order: dataset=%s robot=%s",
            dataset_action_keys,
            robot_action_keys,
        )
    invert_action_keys = parse_action_keys(args.invert_action, action_keys)
    log_robot_feature_config(robot)
    logging.info("Robot action keys: %s", robot_action_keys)
    logging.info("State keys used for observation.state: %s", dataset_state_keys or robot_action_keys)
    logging.info("Action keys used for action_tensor mapping: %s", action_keys)
    logging.info(
        "Action transform: scale=%s invert_all=%s invert_keys=%s",
        args.action_scale,
        args.invert_action_all,
        sorted(invert_action_keys),
    )

    logging.info("Connecting robot on port %s", args.robot_port)
    log_memory_status("before robot connect")
    try:
        robot.connect()
    except Exception:
        logging.exception("Robot connection failed")
        raise

    logging.info("Robot connected.")
    log_memory_status("after robot connect")

    try:
        if args.mcp_trigger_tcp:
            run_mcp_trigger_loop(
                args,
                robot,
                policy,
                device,
                policy_cfg.use_amp,
                invert_action_keys,
                action_keys,
                camera_flips,
                dataset_state_keys,
                dataset_action_keys,
            )
        else:
            run_immediate_loop(
                args,
                robot,
                policy,
                device,
                policy_cfg.use_amp,
                invert_action_keys,
                camera_flips,
                dataset_state_keys,
                dataset_action_keys,
            )
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

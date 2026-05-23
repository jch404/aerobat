#!/usr/bin/env python

"""Save a lightweight SO101 follower calibration file.

This intentionally avoids ``python -m lerobot.calibrate`` and does not create a
SO101Follower object, cameras, policies, torch, or CUDA contexts. It only opens
the Feetech motor bus, reads raw motor registers, and writes the calibration JSON
that LeRobot expects for SO101Follower.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path


for env_name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(env_name, "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from lerobot.motors import Motor, MotorNormMode  # noqa: E402
from lerobot.motors.feetech import FeetechMotorsBus  # noqa: E402


MOTORS = {
    "shoulder_pan": Motor(1, "sts3215", MotorNormMode.RANGE_M100_100),
    "shoulder_lift": Motor(2, "sts3215", MotorNormMode.RANGE_M100_100),
    "elbow_flex": Motor(3, "sts3215", MotorNormMode.RANGE_M100_100),
    "wrist_flex": Motor(4, "sts3215", MotorNormMode.RANGE_M100_100),
    "wrist_roll": Motor(5, "sts3215", MotorNormMode.RANGE_M100_100),
    "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
}

RANGE_MIN = 0
RANGE_MAX = 4095
HALF_TURN_POS = 2047


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read current SO101 follower motor positions and save a lightweight LeRobot calibration JSON."
    )
    parser.add_argument("--port", default="/dev/ttyACM0", help="Feetech controller serial port.")
    parser.add_argument("--id", default="my_follower", help="Robot id used for the calibration filename.")
    parser.add_argument(
        "--calibration-root",
        type=Path,
        default=Path.home() / ".cache/huggingface/lerobot/calibration",
        help="Root LeRobot calibration directory.",
    )
    parser.add_argument(
        "--no-handshake",
        action="store_true",
        help="Open the port without pinging all expected motors first.",
    )
    return parser.parse_args()


def backup_existing(path: Path) -> Path | None:
    if not path.exists():
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = path.with_suffix(path.suffix + f".bak_{timestamp}")
    shutil.copy2(path, backup_path)
    return backup_path


def build_calibration(bus: FeetechMotorsBus) -> tuple[dict[str, dict[str, int]], dict[str, int], dict[str, int]]:
    positions = {name: int(value) for name, value in bus.sync_read("Present_Position", normalize=False).items()}

    existing = bus.read_calibration()
    existing_offsets = {name: int(cal.homing_offset) for name, cal in existing.items()}

    calibration = {}
    for name, motor in MOTORS.items():
        absolute_pos = positions[name] + existing_offsets.get(name, 0)
        calibration[name] = {
            "id": motor.id,
            "drive_mode": 0,
            "homing_offset": int(absolute_pos - HALF_TURN_POS),
            "range_min": RANGE_MIN,
            "range_max": RANGE_MAX,
        }

    return calibration, positions, existing_offsets


def main() -> int:
    args = parse_args()
    save_dir = args.calibration_root / "robots" / "so101_follower"
    save_path = save_dir / f"{args.id}.json"

    bus = FeetechMotorsBus(port=args.port, motors=MOTORS)
    try:
        print(f"Opening {args.port}...")
        bus.connect(handshake=not args.no_handshake)

        calibration, positions, existing_offsets = build_calibration(bus)

        save_dir.mkdir(parents=True, exist_ok=True)
        backup_path = backup_existing(save_path)
        if backup_path is not None:
            print(f"Backed up existing calibration to: {backup_path}")

        with open(save_path, "w") as f:
            json.dump(calibration, f, indent=4)
            f.write("\n")

        print("Raw Present_Position:")
        print(json.dumps(positions, indent=4))
        print("Existing motor Homing_Offset used for absolute-position estimate:")
        print(json.dumps(existing_offsets, indent=4))
        print(f"Saved calibration to: {save_path}")
        print("Saved file contents:")
        print(save_path.read_text())
        return 0
    finally:
        if bus.is_connected:
            bus.disconnect(disable_torque=False)


if __name__ == "__main__":
    if "torch" in sys.modules:
        raise RuntimeError("torch was imported unexpectedly; this script must stay lightweight.")
    raise SystemExit(main())

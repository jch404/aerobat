import json
from pathlib import Path

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus


robot_id = "my_follower"
port = "/dev/ttyACM0"

motors = {
    "shoulder_pan": Motor(1, "sts3215", MotorNormMode.DEGREES),
    "shoulder_lift": Motor(2, "sts3215", MotorNormMode.DEGREES),
    "elbow_flex": Motor(3, "sts3215", MotorNormMode.DEGREES),
    "wrist_flex": Motor(4, "sts3215", MotorNormMode.DEGREES),
    "wrist_roll": Motor(5, "sts3215", MotorNormMode.DEGREES),
    "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
}

save_path = (
    Path.home()
    / ".cache/huggingface/lerobot/calibration/robots/so101_follower"
    / f"{robot_id}.json"
)


def main() -> None:
    bus = FeetechMotorsBus(port=port, motors=motors)

    try:
        bus.connect()
        bus.sync_write("Torque_Enable", 0)
        print("Connected. Torque is OFF, so you can move the arm by hand.")
        input("Move the arm to the calibration pose, then press ENTER...")

        positions = {
            name: int(value)
            for name, value in bus.sync_read("Present_Position", normalize=False).items()
        }

        calibration = {}
        for name, motor in motors.items():
            current = positions[name]
            calibration[name] = {
                "id": motor.id,
                "drive_mode": 0,
                "homing_offset": 0,
                "range_min": max(0, current - 2048),
                "range_max": min(4095, current + 2048),
            }

        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(calibration, f, indent=2)
            f.write("\n")

        print("Present_Position:")
        print(json.dumps(positions, indent=2))
        print(f"Saved calibration to: {save_path}")
    finally:
        if bus.is_connected:
            bus.disconnect()


if __name__ == "__main__":
    main()

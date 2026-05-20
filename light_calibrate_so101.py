import json
import os
from pathlib import Path

from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig

robot_id = "my_follower"
port = "/dev/ttyACM0"

cfg = SO101FollowerConfig(
    port=port,
    id=robot_id,
    cameras={},
)

robot = SO101Follower(cfg)
robot.connect()

print("연결 성공")
print("현재 모터 위치를 읽습니다.")
print("로봇팔을 중간 자세로 놓고 ENTER")
input()

pos = robot.bus.sync_read("Present_Position")
print("현재 위치:", pos)

calib = {}
for name, value in pos.items():
    calib[name] = {
        "id": robot.bus.motors[name].id,
        "drive_mode": 0,
        "homing_offset": 0,
        "range_min": int(value) - 2048,
        "range_max": int(value) + 2048,
    }

save_dir = Path.home() / ".cache/huggingface/lerobot/calibration/robots/so101_follower"
save_dir.mkdir(parents=True, exist_ok=True)

save_path = save_dir / f"{robot_id}.json"

with open(save_path, "w") as f:
    json.dump(calib, f, indent=2)

print("저장 완료:")
print(save_path)

robot.disconnect()

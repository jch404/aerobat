import json
from pathlib import Path
from collections import defaultdict

from lerobot.robots.so101_follower import (
    SO101Follower,
    SO101FollowerConfig,
)

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
print("로봇팔을 중간 자세로 놓고 ENTER")
input()

print("모든 관절을 천천히 끝까지 움직이세요.")
print("앞/뒤/위/아래 전체 범위를 충분히 움직인 뒤 ENTER")
print("기록 중...")

mins = defaultdict(lambda: 4095)
maxs = defaultdict(lambda: 0)

try:
    while True:
        pos = robot.bus.sync_read("Present_Position")

        for name, value in pos.items():
            value = int(value)

            if value < mins[name]:
                mins[name] = value

            if value > maxs[name]:
                maxs[name] = value

except KeyboardInterrupt:
    pass

print("\n결과:\n")

calib = {}

for name in mins.keys():
    rmin = max(0, mins[name])
    rmax = min(4095, maxs[name])

    calib[name] = {
        "id": robot.bus.motors[name].id,
        "drive_mode": 0,
        "homing_offset": 0,
        "range_min": rmin,
        "range_max": rmax,
    }

    print(
        f"{name:15s} | "
        f"MIN={rmin:4d} | "
        f"MAX={rmax:4d} | "
        f"RANGE={rmax-rmin}"
    )

save_dir = (
    Path.home()
    / ".cache/huggingface/lerobot/calibration/robots/so101_follower"
)

save_dir.mkdir(parents=True, exist_ok=True)

save_path = save_dir / f"{robot_id}.json"

with open(save_path, "w") as f:
    json.dump(calib, f, indent=2)

print("\n저장 완료:")
print(save_path)

robot.disconnect()
print("DONE")

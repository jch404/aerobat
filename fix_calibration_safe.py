import json
from pathlib import Path

path = Path.home() / ".cache/huggingface/lerobot/calibration/robots/so101_follower/my_follower.json"

with open(path) as f:
    data = json.load(f)

safe_ranges = {
    "shoulder_pan":  {"range_min": 0, "range_max": 4095},
    "shoulder_lift": {"range_min": 0, "range_max": 4095},
    "elbow_flex":    {"range_min": 0, "range_max": 4095},
    "wrist_flex":    {"range_min": 0, "range_max": 4095},
    "wrist_roll":    {"range_min": 0, "range_max": 4095},
    "gripper":       {"range_min": 0, "range_max": 4095},
}

for name, r in safe_ranges.items():
    if name in data:
        data[name]["range_min"] = r["range_min"]
        data[name]["range_max"] = r["range_max"]

with open(path, "w") as f:
    json.dump(data, f, indent=2)

print("fixed full safe calibration:")
print(path)
print(json.dumps(data, indent=2))

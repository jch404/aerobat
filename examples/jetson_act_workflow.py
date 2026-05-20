#!/usr/bin/env python3
import json
import subprocess
import sys
from pathlib import Path


def run_command(command, check=True):
    print(f"Running: {' '.join(command)}")
    result = subprocess.run(command, check=check)
    return result


def main():
    repo_root = Path(__file__).resolve().parent.parent
    policy_path = "/home/user/project/lerobot/outputs/train/act_floor1_stack/checkpoints/080000/pretrained_model"
    robot_port = "/dev/ttyACM0"
    dataset_root = "/tmp/eval_robot"
    dataset_repo_id = "test/eval_robot"
    task = "test"
    robot_id = "my_follower"
    robot_type = "so101_follower"

    follower_cam_index = 2
    top_cam_index = 0

    print("1) OpenCV 카메라 목록 확인")
    run_command([sys.executable, "-m", "lerobot.find_cameras", "opencv"])

    print("2) 더미 ACT inference 확인")
    act_inference_script = str(repo_root / "examples" / "act_inference.py")
    run_command([
        sys.executable,
        act_inference_script,
        "--policy-path",
        policy_path,
        "--policy-device",
        "cuda",
        "--policy-use-amp",
    ])

    print("3) 실제 로봇 레코딩 실행")
    camera_config = {
        "follower": {
            "type": "opencv",
            "index_or_path": follower_cam_index,
            "width": 640,
            "height": 480,
            "fps": 30,
        },
        "top": {
            "type": "opencv",
            "index_or_path": top_cam_index,
            "width": 640,
            "height": 480,
            "fps": 30,
        },
    }
    camera_config_str = json.dumps(camera_config)

    run_command(
        [
            sys.executable,
            "-m",
            "lerobot.record",
            f"--robot.type={robot_type}",
            f"--robot.port={robot_port}",
            f"--robot.id={robot_id}",
            f"--robot.cameras={camera_config_str}",
            f"--policy.path={policy_path}",
            "--policy.device=cuda",
            f"--dataset.root={dataset_root}",
            f"--dataset.repo_id={dataset_repo_id}",
            f"--dataset.single_task={task}",
            "--dataset.num_episodes=1",
            "--dataset.push_to_hub=false",
            "--dataset.video=false",
            "--dataset.num_image_writer_processes=0",
            "--dataset.num_image_writer_threads_per_camera=1",
            "--display_data=false",
        ]
    )


if __name__ == "__main__":
    main()

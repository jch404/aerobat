#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path


def run_command(command, check=True):
    print(f"Running: {' '.join(command)}")
    subprocess.run(command, check=check)


def main():
    policy_path = "/home/user/project/lerobot/outputs/train/act_floor1_stack/checkpoints/080000/pretrained_model"
    robot_port = "/dev/ttyACM0"
    dataset_root = "/tmp/eval_robot"
    dataset_repo_id = "test/eval_robot"
    task = "test"
    robot_id = "my_follower"
    robot_type = "so101_follower"

    print("1) Running dummy ACT inference on GPU...")
    run_command(
        [
            sys.executable,
            "examples/act_inference.py",
            "--policy-path",
            policy_path,
            "--policy-device",
            "cuda",
            "--policy-use-amp",
        ]
    )

    print("2) Starting robot record with ACT policy on GPU...")
    run_command(
        [
            sys.executable,
            "-m",
            "lerobot.record",
            f"--robot.type={robot_type}",
            f"--robot.port={robot_port}",
            f"--robot.id={robot_id}",
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

#!/usr/bin/env python

from __future__ import annotations

import time

import cv2

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
from lerobot.teleoperators.so101_leader import SO101Leader, SO101LeaderConfig
from lerobot.utils.robot_utils import busy_wait


FOLLOWER_PORT = "/dev/ttyACM0"
LEADER_PORT = "/dev/ttyACM1"
FOLLOWER_ID = "my_follower"
LEADER_ID = "my_leader"

FRONT_CAMERA_INDEX = 0
TOP_CAMERA_INDEX = 2
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

CONTROL_FPS = 30


def make_follower() -> SO101Follower:
    return SO101Follower(
        SO101FollowerConfig(
            port=FOLLOWER_PORT,
            id=FOLLOWER_ID,
            cameras={
                "front": OpenCVCameraConfig(
                    index_or_path=FRONT_CAMERA_INDEX,
                    width=CAMERA_WIDTH,
                    height=CAMERA_HEIGHT,
                    fps=CAMERA_FPS,
                ),
                "top": OpenCVCameraConfig(
                    index_or_path=TOP_CAMERA_INDEX,
                    width=CAMERA_WIDTH,
                    height=CAMERA_HEIGHT,
                    fps=CAMERA_FPS,
                ),
            },
        )
    )


def make_leader() -> SO101Leader:
    return SO101Leader(
        SO101LeaderConfig(
            port=LEADER_PORT,
            id=LEADER_ID,
        )
    )


def show_camera(name: str, rgb_image) -> None:
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    cv2.imshow(name, bgr_image)


def disconnect_leader(leader: SO101Leader) -> None:
    if leader.bus.is_connected:
        leader.disconnect()


def disconnect_follower(follower: SO101Follower) -> None:
    for camera in follower.cameras.values():
        if camera.is_connected:
            camera.disconnect()
    if follower.bus.is_connected:
        follower.bus.disconnect()


def main() -> int:
    leader = make_leader()
    follower = make_follower()

    try:
        print("Connecting leader arm...")
        leader.connect()
        print(f"Leader connected: {LEADER_PORT}")

        print("Connecting follower arm and cameras...")
        follower.connect()
        print(f"Follower connected: {FOLLOWER_PORT}")
        print(f"Front camera: index {FRONT_CAMERA_INDEX}")
        print(f"Top camera: index {TOP_CAMERA_INDEX}")
        print("Teleoperation started. Press q or ESC in the camera window to quit.")

        while True:
            loop_start = time.perf_counter()

            action = leader.get_action()
            observation = follower.get_observation()
            follower.send_action(action)

            show_camera("front", observation["front"])
            show_camera("top", observation["top"])

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

            dt_s = time.perf_counter() - loop_start
            busy_wait(max(0.0, 1.0 / CONTROL_FPS - dt_s))
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        disconnect_follower(follower)
        disconnect_leader(leader)
        print("Teleoperation stopped.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

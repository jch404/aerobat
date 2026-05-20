import time
from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig

cfg = SO101FollowerConfig(
    port="/dev/ttyACM0",
    id="my_follower",
    cameras={},
)

robot = SO101Follower(cfg)
robot.connect()

pos = robot.bus.sync_read("Present_Position")
print("current:", pos)

target = pos.copy()
target["gripper"] = 40.0

print("target:", target)
robot.bus.sync_write("Goal_Position", target)

time.sleep(2)

pos2 = robot.bus.sync_read("Present_Position")
print("after:", pos2)

robot.disconnect()

import time
from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig

cfg = SO101FollowerConfig(port="/dev/ttyACM0", id="my_follower", cameras={})
robot = SO101Follower(cfg)
robot.connect()

for value in [30.0, 70.0, 50.0]:
    pos = robot.bus.sync_read("Present_Position")
    target = pos.copy()
    target["gripper"] = value
    print("move gripper to", value)
    robot.bus.sync_write("Goal_Position", target)
    time.sleep(2)
    print(robot.bus.sync_read("Present_Position"))

robot.disconnect()

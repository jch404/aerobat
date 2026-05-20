from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig

cfg = SO101FollowerConfig(
    port="/dev/ttyACM0",
    id="my_follower",
    cameras={},
)

robot = SO101Follower(cfg)
robot.connect()
print("robot connected ok")
print(robot.bus.sync_read("Present_Position"))
robot.disconnect()

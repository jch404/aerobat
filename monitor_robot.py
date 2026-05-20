import time
from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig

cfg = SO101FollowerConfig(
    port="/dev/ttyACM0",
    id="my_follower",
    cameras={},
)

robot = SO101Follower(cfg)
robot.connect()

print("connected")

try:
    while True:
        pos = robot.bus.sync_read("Present_Position")
        print(pos)
        time.sleep(0.2)

except KeyboardInterrupt:
    pass

robot.disconnect()

import subprocess

cmd = [
    "lerobot-find-motors",
    "--robot.type=so101_follower",
    "--robot.port=/dev/ttyACM0",
    "--robot.id=my_follower",
]

print("실행 명령어:")
print(" ".join(cmd))

result = subprocess.run(cmd)

print("종료 코드:", result.returncode)

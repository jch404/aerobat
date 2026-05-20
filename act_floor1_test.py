import os
import subprocess

# GPU / CPU 부하 제한
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

cmd = [
    "lerobot-record",

    "--robot.type=so101_follower",
    "--robot.port=/dev/ttyACM0",
    "--robot.id=my_follower",

    # 카메라 1개 + 더 낮은 해상도/FPS
    "--robot.cameras={ front: {type: opencv, index_or_path: 2, width: 160, height: 120, fps: 10}}",

    # ACT 모델 경로
    "--policy.path=/home/user/project/lerobot/outputs/train/act_floor1_stack/checkpoints/080000/pretrained_model",

    # 평가 데이터셋
    "--dataset.root=/home/user/project/lerobot/eval_floor1",
    "--dataset.repo_id=cjfgn/eval_floor1",
    "--dataset.single_task=stack_box_floor1",
    "--dataset.num_episodes=1",

    # 저장/업로드 부하 줄이기
    "--dataset.push_to_hub=false",
    "--dataset.video=false",

    # 화면 출력 OFF
    "--display_data=false",
]

print("ACT 추론 실행 시작")
print(" ".join(cmd))

result = subprocess.run(cmd)

print("return code:", result.returncode)
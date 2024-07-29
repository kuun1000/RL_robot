import gymnasium as gym
import panda_gym
from stable_baselines3 import PPO
from panda_gym.pybullet import PyBullet

# 위에서 정의한 커스텀 환경을 등록합니다.
import register_custom_env  # register_custom_env.py 파일을 불러와서 환경을 등록합니다.

# 커스텀 환경 생성
sim = PyBullet()
env = gym.make(
    'CustomRobotPickAndPlace-v5',
    sim=sim
    # render_mode="human",
    # reward_type="dense",  # "dense" or "sparse"
    # control_type="ee",  # "ee" or "joints"
)

# PPO 모델 설정
model = PPO(
    "MlpPolicy",  # 또는 적절한 정책 네트워크 선택
    env,
    verbose=1,
    tensorboard_log="./runs",
    learning_rate=0.001,
)

# 에이전트 학습
model.learn(total_timesteps=200000)

# 학습된 모델 저장
model.save("CustomRobotPickAndPlace_model")

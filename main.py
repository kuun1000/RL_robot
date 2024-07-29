import gymnasium as gym
from stable_baselines3 import DDPG
import environment_registeration

# 환경 생성
env = gym.make('xArmReach-v0', render_mode='human')

# DDPG 모델 생성
model = DDPG(policy="MultiInputPolicy", env=env)

# 모델 학습
model.learn(30_000)
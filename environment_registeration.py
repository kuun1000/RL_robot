import gymnasium as gym
from gymnasium.envs.registration import register

# 환경 등록
register(
    id='xArmReach-v0',
    entry_point='xArmTaskEnv:MyRobotTaskEnv', 
    max_episode_steps=50,
)
import gymnasium as gym
import panda_gym
from gymnasium import spaces
import numpy as np
from gymnasium.envs.registration import register

class CustomPandaPickandPlaceEnv(gym.Env):
    def __init__(self):
        super(CustomPandaPickandPlaceEnv, self).__init__()
        
        # PandaPickandPlace-v3 환경 불러오기
        self.env = gym.make('PandaPickAndPlace-v3', render_mode="human", reward_type="dense", control_type="ee")
        
        # 행동 공간 (action space)
        self.action_space = self.env.action_space
        print(f"action space: {self.action_space}")
        
        # 관측 공간 (observation space)
        self.observation_space = self.env.observation_space
        print(f"observation space: {self.observation_space}")

    def reset(self):
        # 환경 초기화 및 초기 상태 반환
        observation = self.env.reset()
        return observation

    def step(self, action):
        # 주어진 행동을 환경에 적용하고 새로운 상태, 보상, 종료 여부 반환
        observation, reward, done, info = self.env.step(action)
        return observation, reward, done, info

    def render(self, mode='human'):
        # 시각화
        return self.env.render(mode)

    def close(self):
        # 환경 종료 시 정리 작업
        self.env.close()

# # 환경 등록
# register(
#     id='CustomPandaPickandPlace-v0',
#     entry_point='__main__:CustomPandaPickandPlaceEnv',
# )

# # 환경 사용 예시
# if __name__ == "__main__":
#     env = gym.make('CustomPandaPickandPlace-v0')
#     observation = env.reset()
    
#     for _ in range(1000):
#         action = env.action_space.sample()
#         observation, reward, done, info = env.step(action)
#         if done:
#             observation = env.reset()
#         env.render()
    
#     env.close()
CustomPandaPickandPlaceEnv()
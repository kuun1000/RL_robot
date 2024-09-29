import gymnasium as gym
from gymnasium import spaces
import numpy as np


class Grid3DEnv(gym.Env):
    def __init__(self, grid_size=(10, 10, 10), random_goal=False):
        super(Grid3DEnv, self).__init__()

        # 3D 그리드 공간 크기 및 초기 상태 설정
        self.grid_size = grid_size
        self.state = np.zeros(grid_size)
        self.random_goal = random_goal  # 목표 위치를 랜덤하게 설정할지 여부

        # 관찰 공간: 에이전트의 현재 위치 (x, y, z)
        self.observation_space = spaces.Box(low=0, high=max(grid_size) - 1, shape=(3,), dtype=np.int32)

        # 행동 공간: 3D 공간 내 이동 가능 행동 (위, 아래, 왼쪽, 오른쪽, 앞, 뒤)
        self.action_space = spaces.Discrete(6)  # 6개의 3D 이동 행동

        # 시작 위치 및 목표 위치 설정
        self.agent_pos = np.array([0, 0, 0])  # 초기 위치
        self.goal_pos = np.array([grid_size[0] - 1, grid_size[1] - 1, grid_size[2] - 1])  # 기본 목표 위치

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        # 시드 설정 (재현성 보장)
        super().reset(seed=seed)

        # 에이전트의 위치를 초기 위치로 리셋
        self.agent_pos = np.array([0, 0, 0])

        # 목표 위치를 랜덤하게 설정할지 결정
        if self.random_goal:
            self.goal_pos = np.array([np.random.randint(0, self.grid_size[0]),
                                      np.random.randint(0, self.grid_size[1]),
                                      np.random.randint(0, self.grid_size[2])])
        # 초기 상태 반환 및 추가 정보 (info) 제공
        return self.agent_pos, {}

    def step(self, action):
        # 에이전트의 위치를 3D 공간 내에서 이동 (각 행동에 맞는 위치 조정)
        if action == 0 and self.agent_pos[0] < self.grid_size[0] - 1:  # 오른쪽 이동
            self.agent_pos[0] += 1
        elif action == 1 and self.agent_pos[0] > 0:  # 왼쪽 이동
            self.agent_pos[0] -= 1
        elif action == 2 and self.agent_pos[1] < self.grid_size[1] - 1:  # 앞쪽 이동
            self.agent_pos[1] += 1
        elif action == 3 and self.agent_pos[1] > 0:  # 뒤쪽 이동
            self.agent_pos[1] -= 1
        elif action == 4 and self.agent_pos[2] < self.grid_size[2] - 1:  # 위쪽 이동
            self.agent_pos[2] += 1
        elif action == 5 and self.agent_pos[2] > 0:  # 아래쪽 이동
            self.agent_pos[2] -= 1

        # 목표에 도달했는지 여부 확인
        if np.array_equal(self.agent_pos, self.goal_pos):
            reward = 1.0  # 목표 도달 시 보상
            terminated = True
        else:
            reward = -0.1  # 각 스텝마다 작은 패널티
            terminated = False

        # 최대 스텝에 도달하여 강제 종료되는 경우를 위한 `truncated` 변수 (현재 코드에서는 사용하지 않음)
        truncated = False  # 기본적으로 최대 스텝 종료는 없음

        # 현재 상태, 보상, 종료 여부, 추가 정보 반환
        return self.agent_pos, reward, terminated, truncated, {}

    def render(self, mode='human'):
        # 현재 에이전트의 위치 출력
        grid = np.zeros(self.grid_size)
        grid[tuple(self.agent_pos)] = 1  # 에이전트의 위치를 1로 표시
        print(f"Agent's position:\n{self.agent_pos}")

    def close(self):
        pass


# 환경 테스트
if __name__ == "__main__":
    env = Grid3DEnv(grid_size=(5, 5, 5), random_goal=True)  # 목표를 랜덤하게 설정하는 5x5x5 그리드
    obs, info = env.reset()
    print("Initial Observation:", obs)

    done = False
    while not done:
        action = env.action_space.sample()  # 무작위 행동 선택
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Action: {action}, New Observation: {obs}, Reward: {reward}, Terminated: {terminated}")
        done = terminated or truncated

    env.close()

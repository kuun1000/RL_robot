import gymnasium as gym
from gymnasium import spaces

import pybullet as p
import pybullet_data

import numpy as np



class xArm6GraspEnv(gym.Env):
    def __init__(self):
        super(xArm6GraspEnv, self).__init__()

        # PyBullet 연결
        self.client = p.connect(p.GUI)

        
        # 환경 초기화
        self.reset()



    def reset(self):
        # PyBullet 초기화
        p.resetSimulation(self.client)
        p.setGravity(0, 0, -9.8)

        # 테이블 및 로봇 로드
        self.table_id = p.loadURDF("table/table.urdf", basePosition=[0, 0, 0], physicsClientId=self.client)
        self.robot_id = p.loadURDF("lite_6_robotarm.urdf", basePosition=[0, 0, 0.75])
        
        # 큐브 초기화
        self.cube_id = None
        self.reset_cube()



    def reset_cube(self):
        if self.cube_id is not None:
            p.removeBody(self.cube_id, self.client)

        pos = [np.random.uniform(-0.2, 0.2), np.random.uniform(-0.2, 0.2), 0.75]
        orientation = p.getQuaternionFromEuler([0, 0, 0])
        self.cube_id = p.loadURDF("cube_small.urdf", pos, orientation, physicsClientId=self.client)

    def _get_observation(self):
        pass

    def step(self, action):
        pass

    def _compute_reward(self, observation):
        pass

    def _is_done(self, observation):
        pass

    def render(self, mode='human'):
        pass

    def close(self):
        pass
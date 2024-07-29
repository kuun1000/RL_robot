import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data



class xArmEnv(gym.Env):
    def __init__(self):
        super(xArmEnv, self).__init__()
        
        # RGB-D 이미지 크기
        self.height = 480
        self.width = 640

        # 행동 공간 정의: 
        self.action_space = spaces.Discrete(5)

        # 관찰 공간 정의: RGB 이미지, Depth 이미지, 관절 각도
        self.observation_space = spaces.Dict({
            'rgb': spaces.Box(low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8),
            'depth': spaces.Box(low=0, high=255, shape=(self.height, self.width, 1), dtype=np.uint8),
            'joint_angles': spaces.Box(low=-np.pi, high=np.high, shape=(6, ), dtype=np.float32)
        })

        self.np_random, _ = gym.utils.seeding.np_random()

        self.client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.robot = None
        self.target_object = None


    def step(self, action):
        pass

    def reset(self):
        p.resetSimulation(self.client)
        p.setGravity(0, 0, -9.8)

        p.loadURDF("plane.urdf")
        self.robot = p.loadURDF("lite_6_robotarm.urdf", basePosition=[0, 0, 0], useFixedBase=True, physicsClientId=self.client)
        self.table = p.loadURDF("table/table.urdf", basePosition=[0, 0, 0])
        self.target = p.loadURDF("cube_small.urdf", basePosition=[0.5, 0, 0.5])

    def render(self):
        pass

    def close(self):
        p.disconnect(self.client)

    def seed(self, seed=None):
        pass

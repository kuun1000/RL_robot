import gymnasium as gym
from gymnasium import spaces

import pybullet as p
import pybullet_data

import numpy as np



class xArm6GraspEnv(gym.Env):
    def __init__(self):
        super(xArm6GraspEnv, self).__init__()

        # 관찰 공간 정의: RGB-D image, joint angles
        self.height, self.width, self.channel = 640, 480, 4  # TODO: 크기 조절해야 함
        self.num_joints = 7

        self.observation_space = spaces.Dict({
            'rgbd': spaces.Box(low=0, high=255, shape=(self.height, self.width, self.depth), dtype=np.uint8),
            'joint_angles': spaces.Box(low=-np.pi, high=np.pi, shape=(self.num_joints, ), dtype=np.float32)
        })

        # 행동 공간 정의: End-effector displacement(x, y, z), rotation(z), gripper action(closing)
        self.action_space = spaces.Dict({
            'ee_position': spaces.Box(low=np.array([-1.0, -1.0, -1.0]), high=np.array([1.0, 1.0, 1.0]), dtype=np.float32),
            'ee_rotation': spaces.Box(low=np.array([-np.pi]), high=np.array([np.pi]), dtype=np.float32),
            'gripper_action': spaces.Box(low=0.0, high=1.0, shape=(1, ), dtype=np.float32),
        })

        self.client = None
        self.reset()



    def reset(self):
        if self.client is None:
            # PyBullet 초기화   
            p.resetSimulation(self.client)
            p.setAddictionalSearchPath(pybullet_data.getDataPath())

        p.resetSimulation(self.client)
        p.setGravity(0, 0, -9.8, self.client)

        # 테이블 및 로봇 로드
        self.table_id = p.loadURDF("table/table.urdf", basePosition=[0, 0, 0], physicsClientId=self.client)
        self.robot_id = p.loadURDF("lite_6_robotarm.urdf", basePosition=[0, 0, 0.75])
        
        # 큐브 초기화
        self.cube_id = None
        self.reset_cube()

        return self._get_observation()



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
import gymnasium as gym
from gymnasium import spaces

import pybullet as p
import pybullet_data

import numpy as np
import cv2
from typing import Any, Dict, Iterator, Optional


class xArmEnv(gym.Env):
    def __init__(self):
        super(xArmEnv, self).__init__()

        # 관찰 공간 정의: RGB image, Depth image, Joint state
        
        self.height, self.width, self.channel = 480, 640, 3     # 이미지 해상도 설정 -> TODO: 해상도 조절 필요
        self.num_joints = 6     # End-effector 제외한 관절 개수

        self.observation_space = spaces.Dict({
            'rgb': spaces.Box(low=0, high=255, shape=(self.height, self.width, self.channel), dtype=np.uint8),
            'depth': spaces.Box(low=0, high=255, shape=(self.height, self.width, 1), dtype=np.uint8),
            'joint_state': spaces.Box(low=-np.pi, high=np.pi, shape=(self.num_joints, ), dtype=np.float32)
        })

        # 행동 공간 정의: End-effector displacement(x, y, z), gripper action(closing)
        self.action_space = spaces.Discrete(8)

        self.client = None
        self.reset()



    def reset(self):
        if self.client is None:
            # PyBullet 초기화   
            self.client = p.connect(p.GUI)
            p.setAdditionalSearchPath(pybullet_data.getDataPath())

        p.resetSimulation(self.client)
        p.setGravity(0, 0, -9.8, self.client)

        # 테이블 로드
        self.table_id = p.loadURDF("table/table.urdf", 
                                   basePosition=[0, 0, 0], 
                                   baseOrientation=p.getQuaternionFromEuler([0, 0, np.pi/2]), 
                                   physicsClientId=self.client)
        
        # 로봇 로드 및 위치 수정
        self.robot_id = p.loadURDF("lite_6_robotarm_revise.urdf", 
                                   basePosition=[-0.45, 0.00, 0.65], 
                                   baseOrientation=p.getQuaternionFromEuler([0, 0, -np.pi/2]),
                                   useFixedBase=True)
        
        # 충돌 설정
        p.setCollisionFilterPair(self.robot_id, self.table_id, -1, -1, 1)
        p.setCollisionFilterPair(self.robot_id, self.robot_id, -1, -1, 1)
        
        self.ee = 6
        self.camera = 9

        # 큐브 초기화
        self.cube_id = None
        self.target_pos = None
        self.create_cube()

        # 스텝 수 초기화
        self.step_count = 0

        return self.get_observation()



    def create_cube(self):
        # 테이블 범위 파악
        aabb_min, aabb_max = p.getAABB(self.table_id)
        
        # 큐브 생성
        pos1 = [np.random.uniform(aabb_min[0], aabb_max[0]), 
                np.random.uniform(aabb_min[1], aabb_max[1]), 
                aabb_max[2] + 0.03]
        
        ori1 = p.getQuaternionFromEuler([0, 0, 0])

        self.cube_id = p.loadURDF("cube_small.urdf", pos1, ori1, useFixedBase=False, physicsClientId=self.client)
        
        # 목표 위치 시각화
        while True:
            pos2 = [np.random.uniform(aabb_min[0], aabb_max[0]), 
                    np.random.uniform(aabb_min[1], aabb_max[1]), 
                    aabb_max[2]]
            if not np.allclose(pos1[:2], pos2[:2], atol=0.1):  # 큐브와 목표 위치가 같지 않도록 설정
                break

        self.target_position = pos2

        cube_aabb_min, cube_aabb_max = p.getAABB(self.cube_id)
        cube_length = (cube_aabb_max[0] - cube_aabb_min[0]) / 2

        p1 = [pos2[0] - cube_length, pos2[1] - cube_length, pos2[2]]
        p2 = [pos2[0] + cube_length, pos2[1] - cube_length, pos2[2]]
        p3 = [pos2[0] + cube_length, pos2[1] + cube_length, pos2[2]]
        p4 = [pos2[0] - cube_length, pos2[1] + cube_length, pos2[2]]

        p.addUserDebugLine(p1, p2, [1, 0, 0], 2)
        p.addUserDebugLine(p2, p3, [1, 0, 0], 2)
        p.addUserDebugLine(p3, p4, [1, 0, 0], 2)
        p.addUserDebugLine(p4, p1, [1, 0, 0], 2)



    def arm_camera(self):
        # Center of mass position and orientation(of link-9)
        com_p, com_o, _, _, _, _ = p.getLinkState(self.robot_id, self.camera)

        # Camera setting
        fov, aspect, near, far = 60, self.width/self.height, 0.01, 15
        projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)

        rot_matrix = p.getMatrixFromQuaternion(com_o) 
        rot_matrix = np.array(rot_matrix).reshape(3, 3)

        # Initial vectors
        init_camera_vector = (0, 0, -1)  # z-axis
        init_up_vector = (0, 1, 0)  # y-axis

        # Rotated vectors
        camera_vector = rot_matrix.dot(init_camera_vector)
        up_vector = rot_matrix.dot(init_up_vector)
        view_matrix = p.computeViewMatrix(com_p, com_p + 1.0 * camera_vector, up_vector)

        img = p.getCameraImage(self.width, self.height, view_matrix, projection_matrix,
                               shadow=True,
                               renderer=p.ER_BULLET_HARDWARE_OPENGL)

        rgb_opengl = np.reshape(img[2], (self.height, self.width, 4)) * (1. / 255.)
        rgb_opengl_uint8 = np.array(rgb_opengl * 255, dtype=np.uint8)
        rgb_img = cv2.cvtColor(rgb_opengl_uint8, cv2.COLOR_RGB2BGR)

        depth_opengl = np.reshape(img[3], (self.height, self.width))
        depth_img_normalized = cv2.normalize(depth_opengl, None, 0, 255, cv2.NORM_MINMAX)
        depth_img = np.uint8(depth_img_normalized)

        return rgb_img, depth_img



    def get_observation(self):
        
        # 그리퍼 위치
        gripper_pos = p.getLinkState(self.robot_id, self.ee)[0]
        # 큐브 위치
        cube_pos = p.getBasePositionAndOrientation(self.cube_id)[0]
        # 목표 위치
        target_pos = self.target_pos
        # 큐브 잡혔는지 여부
        cube_grasped = len(p.getContactPoints(bodyA=self.robot_id, bodyB=self.cube_id)) > 0

        observation = {
            'gripper_pos': gripper_pos,
            'cube_pos': cube_pos,
            'target_pos': target_pos,
            'cube_grasped': cube_grasped
        }

        return observation



    def step(self, action):
        # 행동 적용
        self.apply_action(action)

        # 시뮬레이션 한 스텝 진행
        p.stepSimulation()

        # 새로운 상태 관찰
        observation = self.get_observation()

        # 보상 계산
        reward = self.compute_reward(observation)

        # 에피소드 종료 여부 판단
        done = self.is_done(observation, self.step_count, max_steps=200)

        # 스텝 수 증가
        self.step_count += 1
        
        # 추가 정보 TODO: 이후 디버깅 또는 학습 분석을 위한 추가 정보
        info = {}

        return observation, reward, done, info
    


    def compute_reward(self, observation):
        # observation 추출
        gripper_pos = observation['gripper_pos']   # 그리퍼의 현재 위치
        cube_pos = np.array(observation['cube_pos'])     # 물체의 현재 위치
        target_pos = np.array(observation['target_pos'])     # 목표 위치
        cube_grasped = observation['cube_grasped']  # 물체가 잡혔는지 여부

        # 거리 기반 보상 
        assert cube_pos.shape == target_pos.shape
        dist = np.linalg.norm(cube_pos - target_pos, axis=-1)
        dist_reward = -np.round(dist, 6)

        return dist_reward



    def is_done(self, observation, step_count, max_steps=200):
        cube_pos = observation['cube_pos']
        target_pos = observation['target_pos']

        # 물체가 목표 위치에 정확히 놓였는지
        dist = np.linalg.norm(np.array(cube_pos) - np.array(target_pos))
        if dist < 0.05:
            return True
        
        # 최대 스텝 수 초과
        if step_count >= max_steps:
            return True
        
        # (선택) 물체가 테이블 아래로 떨어졌는지 
        # if cube_pos[2] < 테이블 z좌표 - 0.05:
        #     return True
    
        return False



    def render(self, mode='human'):
        self.arm_camera()


    def close(self):
        if self.client is not None:
            p.disconnect(self.client)
            self.client = None
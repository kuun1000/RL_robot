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

        # 행동 공간 정의: End-effector displacement(x, y, z), rotation(z), gripper action(closing)
        self.num_pos_actions = 5  # 예시
        self.num_rot_actions = 4  # 예시
        self.num_grip_actions = 2  # open/close 2개
        
        self.action_space = spaces.Dict({
            'end_effector_position': spaces.MultiDiscrete([self.num_pos_actions] * 3),
            'end_effector_rotation': spaces.Discrete(self.num_rot_actions),
            'gripper_action': spaces.Discrete(self.num_grip_actions),
        })

        self.client = None
        self.reset()



    def reset(self):
        if self.client is None:
            # PyBullet 초기화   
            self.client = p.connect(p.GUI)
            p.setAdditionalSearchPath(pybullet_data.getDataPath())

        p.resetSimulation(self.client)
        p.setGravity(0, 0, -9.8, self.client)

        # 테이블 및 로봇 로드
        self.table_id = p.loadURDF("table/table.urdf", 
                                   basePosition=[0, 0, 0], 
                                   baseOrientation=p.getQuaternionFromEuler([0, 0, np.pi/2]), 
                                   physicsClientId=self.client)
        robot_base_position = [-0.45, 0.00, 0.60]
        robot_base_orientation = p.getQuaternionFromEuler([0, 0, -np.pi/2])
        self.robot_id = p.loadURDF("lite_6_robotarm.urdf", basePosition=robot_base_position, baseOrientation=robot_base_orientation, useFixedBase=True)
        
        p.setCollisionFilterPair(self.robot_id, self.table_id, -1, -1, 1)
        p.setCollisionFilterPair(self.robot_id, self.robot_id, -1, -1, 1)
        
        self.ee = 6
        self.camera = 9

        # 큐브 초기화
        self.cube_id = None
        self.reset_cube()

        return self._get_observation()


    def reset_cube(self):
        if self.cube_id is not None:
            p.removeBody(self.cube_id, self.client)

        pos = [np.random.uniform(-0.2, 0.1), np.random.uniform(0.05, 0.25), 0.65]
        orientation = p.getQuaternionFromEuler([0, 0, 0])
        self.cube_id = p.loadURDF("cube_small.urdf", pos, orientation, physicsClientId=self.client)


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

        # segmentation_image = np.array(img[4]).reshape((self.height, self.width))
        # seg_img = cv2.applyColorMap(np.uint8(segmentation_image * 255 / segmentation_image.max()), cv2.COLORMAP_JET)

        return rgb_img, depth_img



    def _get_observation(self):
        # RGB-D image
        rgb_img, depth_img = self.arm_camera()
        rgbd_img = np.dstack((rgb_img, depth_img))

        # Joint angles
        joint_states = p.getJointStates(self.robot_id, range(self.num_joints))
        joint_positions = [state[0] for state in joint_states]

        # Cube position
        cube_pos, _ = p.getBasePositionAndOrientation(self.cube_id)

        observation = {
            'rgbd': rgbd_img,
            'joint_angles': np.array(joint_positions),
            'cube_position': np.array(cube_pos)
        }

        return observation



    def step(self, action):
        # Apply action
        pos_delta = np.linspace(-1.0, 1.0, self.num_pos_actions)
        end_effector_pos_delta = np.array([pos_delta[action['end_effector_position'][i]] for i in range(3)])
    
        rot_delta = np.linspace(-np.pi, np.pi, self.num_rot_actions)
        end_effector_rot_delta = rot_delta[action['end_effector_rotation']]

        gripper_action = action['gripper_action']

        # end_effector_pos_delta = action['end_effector_position']
        # end_effector_rot_delta = action['end_effector_rotation']
        # gripper_action = action['gripper_action'][0]

        # Previous position of end-effector
        prev_end_effector_pos = p.getLinkState(self.robot_id, self.ee)[0]

        # Current end-effector position and rotation
        end_effector_pos = p.getLinkState(self.robot_id, self.ee)[0]
        
        # Calculate target position and orientation
        new_pos = np.array(end_effector_pos) + np.array(end_effector_pos_delta)
        new_orn = p.getQuaternionFromEuler([0, 0, end_effector_rot_delta])
        
        # InverseKinematics
        jointPoses = p.calculateInverseKinematics(self.robot_id, self.ee, new_pos, new_orn)
        
        # Control each joints to move end-effector to target position
        for i in range(self.num_joints):
            p.setJointMotorControl2(bodyIndex=self.robot_id,
                                jointIndex=i,
                                controlMode=p.POSITION_CONTROL,
                                targetPosition=jointPoses[i],
                                )

        # Apply gripper action
        gripper_finger_indices = [7, 8]
        for joint_index in gripper_finger_indices:
            p.setJointMotorControl2(self.robot_id, joint_index, p.POSITION_CONTROL, targetPosition=gripper_action)

        p.stepSimulation(self.client)

        # Current position of end-effector
        cur_end_effector_pos = p.getLinkState(self.robot_id, self.ee)[0]

        cube_pos = p.getBasePositionAndOrientation(self.cube_id)[0]
        # print(f"Gripper Position: {cur_end_effector_pos}, Cube Position: {cube_pos}")

        obs = self._get_observation()
        reward = self._compute_reward(obs, prev_end_effector_pos, cur_end_effector_pos, new_pos)
        done = self._is_done(obs)

        return obs, reward, done
    


    def _compute_reward(self, observation, initial_pos, final_pos, target_pos):
        cube_pos = observation['cube_position']

        gripper_contact = p.getContactPoints(bodyA=self.robot_id, bodyB=self.cube_id)
        # print(f'gripper contact: {gripper_contact}')
        reward = 0.0
        
        if np.allclose(final_pos, initial_pos, atol=1e-2):
            reward -= 1.0
        
        else:
            if len(gripper_contact) > 0:
                if cube_pos[2] > 1.0:
                    reward += 10.0
                else:
                    reward += 1.0
        reward = -0.025

        return reward



    def _is_done(self, observation):
        cube_pos = observation['cube_position']
        return cube_pos[2] >1.0
    


    def render(self, mode='human'):
        self.arm_camera()


    def close(self):
        if self.client is not None:
            p.disconnect(self.client)
            self.client = None

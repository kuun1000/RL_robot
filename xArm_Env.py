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

        # 관찰 공간 정의: RGB-D image, joint angles
        self.height, self.width, self.channel = 480, 640, 4  # TODO: 크기 조절해야 함
        self.num_joints = 7

        self.observation_space = spaces.Dict({
            # 'rgbd': spaces.Box(low=0, high=255, shape=(self.height, self.width, self.channel), dtype=np.uint8),
            'joint_angles': spaces.Box(low=-np.pi, high=np.pi, shape=(self.num_joints, ), dtype=np.float32)
        })

        # 행동 공간 정의: End-effector displacement(x, y, z), rotation(z), gripper action(closing)
        self.action_space = spaces.Dict({
            'end_effector_position': spaces.Box(low=np.array([-1.0, -1.0, -1.0]), high=np.array([1.0, 1.0, 1.0]), dtype=np.float32),
            'end_effector_rotation': spaces.Box(low=np.array([-np.pi]), high=np.array([np.pi]), dtype=np.float32),
            'gripper_action': spaces.Box(low=0.0, high=1.0, shape=(1, ), dtype=np.float32),
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
        self.robot_id = p.loadURDF("lite_6_robotarm.urdf", 
                                   basePosition=[-0.45, -0.05, 0.60], 
                                   useFixedBase=True)
        
        p.setCollisionFilterPair(self.robot_id, self.table_id, -1, -1, 1)
        p.setCollisionFilterPair(self.robot_id, self.robot_id, -1, -1, 1)
        
        self.ee = 6
        self.camera = 9

        # 큐브 초기화
        self.cube_object_id = None
        self.cube_target_id = None
        self.create_cube()

        return self._get_observation()



    def create_cube(self):
        # 테이블 범위 파악
        aabb_min, aabb_max = p.getAABB(self.table_id)  # 테이블의 AABB 범위
        
        # 첫 번째 큐브 생성 (ghost=False)
        pos1 = [np.random.uniform(aabb_min[0], aabb_max[0]), 
                np.random.uniform(aabb_min[1], aabb_max[1]), 
                aabb_max[2] + 0.05]  # 테이블 위에 약간 띄워서 배치
        orientation1 = p.getQuaternionFromEuler([0, 0, 0])
        self.cube_object_id = p.loadURDF("cube_small.urdf", pos1, orientation1, useFixedBase=False, globalScaling=1.0, physicsClientId=self.client)
        
        # 두 번째 큐브 생성 (ghost=True)
        while True:
            pos2 = [np.random.uniform(aabb_min[0], aabb_max[0]), 
                    np.random.uniform(aabb_min[1], aabb_max[1]), 
                    aabb_max[2] + 0.05]
            if not np.allclose(pos1, pos2, atol=0.1):  # 두 큐브의 위치가 같지 않도록 설정
                break

        orientation2 = p.getQuaternionFromEuler([0, 0, 0])
        self.cube_target_id = p.loadURDF("cube_small.urdf", pos2, orientation2, useFixedBase=False, globalScaling=1.0, physicsClientId=self.client)
        
        # 두 번째 큐브를 ghost 모드로 설정하여 충돌 무시
        p.changeVisualShape(self.cube_target_id, -1, rgbaColor=[0, 1, 0, 0.5])  # 반투명 초록색으로 설정
        p.setCollisionFilterPair(self.cube_target_id,  self.cube_target_id, -1, -1, enableCollision=0)  # ghost 모드 설정



    def _create_geometry(
            self,
            body_name: str,
            geom_type: int,
            mass: float = 0.0,
            position: Optional[np.ndarray] = None,
            ghost: bool = False,
            lateral_friction: Optional[float] = None,
            spinning_friction: Optional[float] = None,
            visual_kwargs: Dict[str, Any] = {},
            collision_kwargs: Dict[str, Any] = {},
        ) -> None:
            """Create a geometry.

            Args:
                body_name (str): The name of the body. Must be unique in the sim.
                geom_type (int): The geometry type. See self.physics_client.GEOM_<shape>.
                mass (float, optional): The mass in kg. Defaults to 0.
                position (np.ndarray, optional): The position, as (x, y, z). Defaults to [0, 0, 0].
                ghost (bool, optional): Whether the body can collide. Defaults to False.
                lateral_friction (float or None, optional): Lateral friction. If None, use the default pybullet
                    value. Defaults to None.
                spinning_friction (float or None, optional): Spinning friction. If None, use the default pybullet
                    value. Defaults to None.
                visual_kwargs (dict, optional): Visual kwargs. Defaults to {}.
                collision_kwargs (dict, optional): Collision kwargs. Defaults to {}.
            """
            position = position if position is not None else np.zeros(3)
            baseVisualShapeIndex = self.physics_client.createVisualShape(geom_type, **visual_kwargs)
            if not ghost:
                baseCollisionShapeIndex = self.physics_client.createCollisionShape(geom_type, **collision_kwargs)
            else:
                baseCollisionShapeIndex = -1
            self._bodies_idx[body_name] = self.physics_client.createMultiBody(
                baseVisualShapeIndex=baseVisualShapeIndex,
                baseCollisionShapeIndex=baseCollisionShapeIndex,
                baseMass=mass,
                basePosition=position,
            )

            if lateral_friction is not None:
                self.set_lateral_friction(body=body_name, link=-1, lateral_friction=lateral_friction)
            if spinning_friction is not None:
                self.set_spinning_friction(body=body_name, link=-1, spinning_friction=spinning_friction)



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

        segmentation_image = np.array(img[4]).reshape((self.height, self.width))
        seg_img = cv2.applyColorMap(np.uint8(segmentation_image * 255 / segmentation_image.max()), cv2.COLORMAP_JET)

        return rgb_img, depth_img, seg_img



    def _get_observation(self):
        # RGB-D image
        rgb_img, depth_img, seg_img = self.arm_camera()
        rgbd_img = np.dstack((rgb_img, depth_img))

        # Joint angles
        joint_states = p.getJointStates(self.robot_id, range(self.num_joints))
        joint_positions = [state[0] for state in joint_states]

        # Cube position
        cube_pos, _ = p.getBasePositionAndOrientation(self.cube_object_id)

        observation = {
            'rgbd': rgbd_img,
            'joint_angles': np.array(joint_positions),
            'cube_position': np.array(cube_pos)
        }

        return observation



    def step(self, action):
        # Apply action
        end_effector_pos_delta = action['end_effector_position']
        end_effector_rot_delta = action['end_effector_rotation']
        gripper_action = action['gripper_action'][0]

        # Previous position of end-effector
        prev_end_effector_pos = p.getLinkState(self.robot_id, self.ee)[0]

        # Current end-effector position and rotation
        end_effector_pos = p.getLinkState(self.robot_id, self.ee)[0]
        
        # Calculate target position and orientation
        new_pos = np.array(end_effector_pos) + np.array(end_effector_pos_delta)
        new_orn = p.getQuaternionFromEuler([0, 0, end_effector_rot_delta[0]])
        
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

        cube_pos = p.getBasePositionAndOrientation(self.cube_object_id)[0]
        # print(f"Gripper Position: {cur_end_effector_pos}, Cube Position: {cube_pos}")

        obs = self._get_observation()
        reward = self._compute_reward(obs, prev_end_effector_pos, cur_end_effector_pos, new_pos)
        done = self._is_done(obs)

        return obs, reward, done
    


    def _compute_reward(self, observation, initial_pos, final_pos, target_pos):
        cube_pos = observation['cube_position']

        gripper_contact = p.getContactPoints(bodyA=self.robot_id, bodyB=self.cube_object_id)
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

'''
Xarm DRL project RL Environment
'''
import os
import gymnasium as gym
from gymnasium import spaces, logger
from A2C import *
from TD3 import *
import numpy as np
import pybullet as p
from collections import deque
from replay_buffer import ReplayBuffer
import pybullet_data
import cv2
# import utils

class XArmEnv(gym.Env):
    def __init__(self, steps = 500, train=True):
        self.train = train
        
        self.client = None
        self.cube_id = None
        self.robot_id = None
        self.table_id = None
        self.place_pos = None
        # high = np.array([1.0, 1.0, 1.0, 1.0, 10, 10, 10, 10, 1.0, 1.0, 1.0])
        # self.action_space = spaces.Discrete(19)
        # self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        
        # action for joints and grasp on/off => 7
        self.action_space = spaces.Box(-np.pi, np.pi, (7,), dtype=np.float32)
        
        # 11 observation state
        self.observation_space = spaces.Dict({
            'grasp_state': spaces.Discrete(2),  # True: grasp / False: no grasp
            'ee_position_state': spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),  # End-effector position
            'cube_position_state': spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),  # Cube position
            'place_position_state': spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)  # place position
        })

        # self.prev_dist = 0
        self.best_dist = 9999999
        self.ee = 6
        self.camera = 9
        self.nsteps = None
        self.joint_indices = [0,1,2,3,4,5]
        self.gripper_indiecs = [7,8]

        self.state = None
        #---thresholds for episode-----
        self.max_episode_steps = steps
        #--------------------------

        self.steps_beyond_done = None
        
        # self.reset()
        # print(self.reset())

# clear
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

        self.place_pos = pos2

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
# clear
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
# clear
    def render_camera(self, mode='human'):
        self.arm_camera()

    def get_observation(self):
        # 그리퍼 위치
        gripper_pos = p.getLinkState(self.robot_id, self.ee)[0]
        # 큐브 위치
        cube_pos = p.getBasePositionAndOrientation(self.cube_id)[0]
        # 목표 위치
        place_pos = self.place_pos 
        # 큐브 잡혔는지 여부
        cube_grasped = int(len(p.getContactPoints(bodyA=self.robot_id, bodyB=self.cube_id)) >= 2)

        observation = np.array([cube_grasped,
                                gripper_pos[0],gripper_pos[1],gripper_pos[2],
                                cube_pos[0],cube_pos[1],cube_pos[2],
                                place_pos[0],place_pos[1],place_pos[2]],dtype=np.float32)

        return observation

    def step(self, action):
        # Simulation step
        p.stepSimulation()
        self.nsteps += 1
        # gripper_pos = p.getLinkState(self.robot_id, self.ee)[0]

        # print(action)
        # ===================================
        # Add pybullet operation
        # pybullet_action()

        # joint_poses = p.calculateInverseKinematics(bodyUniqueId = self.robot_id, endEffectorLinkIndex = self.ee, targetPosition = action)
        p.setJointMotorControlArray(bodyUniqueId=self.robot_id,
                                    jointIndices = self.joint_indices,
                                    controlMode = p.POSITION_CONTROL,
                                    targetPositions = action[0:6])
        if action[6] >= 0:
            grasp_angle = np.pi / 4
        else:
            grasp_angle = -np.pi / 4
        
        p.setJointMotorControl2(bodyUniqueId=self.robot_id,
                                jointIndex = self.gripper_indiecs[0],
                                controlMode = p.POSITION_CONTROL,
                                targetPosition = grasp_angle)
        p.setJointMotorControl2(bodyUniqueId=self.robot_id,
                                jointIndex = self.gripper_indiecs[1],
                                controlMode = p.POSITION_CONTROL,
                                targetPosition = grasp_angle)
        
        # ee_pos, cube_grasp, grasp = pybullet_informations()
        # ===================================

        observation = self.get_observation()
        # define state same as observation space to match correctly
        self.state = observation
        
        # --------REWARD---------
        if(self.train):
            reward = self.compute_reward(observation)
            done = self.is_done(observation, self.nsteps)
            return self.state, reward, done, {}
        
        # pybullet_action()
        # ee_pos, cube_grasp, grasp = pybullet_informations()
        # XArm_API(data from ee_pos with pybullet API) 
        reward = self.compute_reward(observation)

        info = {} # add something needed
        return self.state, reward, done, info
# clear
    def compute_reward(self, observation):
        cube_pos = observation[4:7]
        place_pos = observation[7:10]
        ee_pos = observation[0:3]
        grasp = observation[0]

        distance = np.linalg.norm(ee_pos - cube_pos)
        print("Best distance:\n",self.best_dist)
        if distance < self.best_dist:
            self.best_dist = distance
            return 10
        else:
            return -1

        # if distance < 0.1:  # distance 범위 알아야함
        #     return 10.0
        # elif distance < 0.5:
        #     return 1.0
        # elif distance < 1.0:
        #     return 0.5
        # else:
        #     return -1.0
# clear
    def is_done(self, observation, step_count):
        cube_pos = observation[4:7]
        place_pos = observation[7:10]

        # 물체가 목표 위치에 정확히 놓였는지
        dist = np.linalg.norm(np.array(cube_pos) - np.array(place_pos))
        if dist < 0.05:
            return True
        
        # 최대 스텝 수 초과
        if step_count >= self.max_episode_steps:
            return True
        
        # (선택) 물체가 테이블 아래로 떨어졌는지 
        # if cube_pos[2] < 테이블 z좌표 - 0.05:
        #     return True
    
        return False
# clear
    def reset(self, init_quat = None):
        if self.client is None:
            # PyBullet 초기화   
            self.client = p.connect(p.GUI)
            p.setAdditionalSearchPath(pybullet_data.getDataPath())

        p.resetSimulation(self.client)
        p.setGravity(0, 0, -9.8, self.client)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.resetSimulation(self.client)
        p.setGravity(0, 0, -9.8, self.client)
        # 로봇 로드 및 위치 수정
        self.robot_id = p.loadURDF("robotarm_revise.urdf", 
                                   basePosition=[-0.45, 0.00, 0.665], #+++
                                   baseOrientation=p.getQuaternionFromEuler([0, 0, -np.pi/2]),
                                   useFixedBase=True)
        self.table_id = p.loadURDF("table/table.urdf", 
                                   basePosition=[0, 0, 0], 
                                   baseOrientation=p.getQuaternionFromEuler([0, 0, np.pi/2]), 
                                   physicsClientId=self.client)
        self.create_cube()
        # 충돌 설정
        p.setCollisionFilterPair(self.robot_id, self.table_id, -1, -1, 1)
        p.setCollisionFilterPair(self.robot_id, self.robot_id, -1, -1, 1)

        observation = self.get_observation()
        # define state same as observation space to match correctly
        self.state = observation

        self.nsteps = 0        
        self.steps_beyond_done = 0
        self.best_dist = 9999999
        # print(np.array(self.state))
        return self.state
# clear
    def close(self):
        if self.client is not None:
            p.disconnect(self.client)
            self.client = None
# clear
    def load_model(self, model_path, device='cuda'):
        """ Load the trained actor model """
        # Determine the state dimension by flattening all components
        state_dim = 1 + 3 * 3  # 1 for grasp_state, 3 for each of ee_position_state, cube_position_state, place_position_state
        action_dim = self.action_space.n
        max_action = 1.0  # Define max action as per the environment requirements

        self.device = device
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor.load_state_dict(torch.load(model_path, map_location=device))
        self.actor.eval()
# clear
    def predict(self, state):
        """ Predict the action based on the current state """
        if not hasattr(self, 'actor'):
            raise ValueError("Model not loaded. Use load_model() to load the trained model.")
        
        # Flatten the state dictionary
        flattened_state = self.flatten_state(state)
        state_tensor = torch.FloatTensor(flattened_state).unsqueeze(0).to(self.device)

        # Predict the action
        action = self.actor(state_tensor).detach().cpu().numpy().flatten()
        return action
# clear
    def train_model(self, episodes=10000, batch_size=64, gamma=0.99, polyak=0.995, policy_noise=0.2, noise_clip=0.5, policy_delay=2, exploration_noise=0.1):
        """ Train the agent using the TD3 algorithm """
        state_dim = 10  # Flattened state dimension
        action_dim = self.action_space.shape[0]  # continous actions
        # print(action_dim)
        max_action = 1.0  # Not strictly necessary for discrete actions
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        save_path = './weight'
        os.makedirs(save_path, exist_ok=True)
        model_save_path = os.path.join(save_path, 'model.pth')

        # Initialize the TD3 agent
        agent = TD3(lr=1e-3, state_dim=state_dim, action_dim=action_dim, max_action=max_action, device=device)

        replay_buffer = ReplayBuffer()

        for episode in range(episodes):
            state = self.reset()
            # print("reset_state_info : ", state)
            done = False
            episode_reward = 0
            step_count = 0

            while not done:
                # Flatten the state for input to the model
                # flattened_state = self.flatten_state(state)
                
                # Select action according to the current policy
                action = agent.select_action(state)
                action = action + np.random.normal(0, exploration_noise, size=self.action_space.shape[0])
                # print(action)
                # action = action.clip(self.action_space.low, self.action_space.high)
                next_state, reward, done, _ = self.step(action)

                # Store experience in the replay buffer
                replay_buffer.add((state, action, reward, next_state, done))

                # Sample a batch of transitions from the replay buffer
                if len(replay_buffer.buffer) > batch_size:
                    state_batch, action_batch, reward_batch, next_state_batch, done_batch = replay_buffer.sample(batch_size)
                    
                    # Convert batch data to tensors
                    state_batch = torch.FloatTensor(state_batch).to(device)
                    action_batch = torch.FloatTensor(action_batch).to(device)
                    reward_batch = torch.FloatTensor(reward_batch).unsqueeze(1).to(device)
                    next_state_batch = torch.FloatTensor(next_state_batch).to(device)
                    done_batch = torch.FloatTensor(done_batch).unsqueeze(1).to(device)
                    
                    # Update the agent
                    agent.update(replay_buffer, n_iter=1, 
                                 batch_size=batch_size, gamma=gamma, 
                                 polyak=polyak, policy_noise=policy_noise, 
                                 noise_clip=noise_clip, 
                                 policy_delay=policy_delay)

                state = next_state
                episode_reward += reward
                step_count += 1

            print(f"Episode {episode + 1}/{episodes}, Reward: {episode_reward}, Steps: {step_count}")


        
        torch.save(agent.actor.state_dict(), model_save_path)
        print(f"Model saved, done training!!!!!!!!!")
    #def predict
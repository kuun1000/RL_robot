import random
import gymnasium as gym
import pybullet as p
import pybullet_data
import numpy as np
import cv2
from gymnasium import spaces
import time

class XarmRobotEnv(gym.Env):
# Initialization ============================================
    def __init__(self, is_for_training = False):
        super(XarmRobotEnv,self).__init__()

        # Camera info
        self.height, self.width, self.channel = 480, 640, 3
        
        # simulation setup
        self.use_gui = not is_for_training
        self.realtime = not is_for_training

        self.num_step_max_per_episode = 1
        self.num_step = 0  
        self.num_simulation_steps = 2000
        self.tolerance = 0.005 # Tolerance value for the movement of the robot
        self.tolerance_reward = 0.05

        # Robot
        # joints : 0~5 + gripper R/L => 8 joints
        '''
        self.joint_velocities =np.array(
                                            [0.0, 0.0, 0.0, 0.0,
                                            0.0, 0.0, 0.0, 0.0])
        '''
        self.joint_angles_des = None
        self.joint_positions_start = np.array(
                                            [0.0, 0.0, 0.0, 0.0,
                                            0.0, 0.0, 0.0, 0.0])
        
        # Quantities for the observation and the agent's objective
        self.desired_cube_position = None

        # Open Pybullet
        try:
            if self.use_gui:
                p.connect(p.GUI)
                p.resetDebugVisualizerCamera(
                    cameraDistance=1.6,
                    cameraYaw=120,
                    cameraPitch=-50
                )
            else:
                p.connect(p.DIRECT)
        except Exception as e:
            print(f"Errore durante la connessione a PyBullet: {e}")
            raise

        # Bring data from pybullet storage
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Spawn ground plane and table
        try:
            self.plane_id = p.loadURDF("plane.urdf")
            self.table_id = p.loadURDF("table/table.urdf", 
                                   basePosition=[0, 0, 0], 
                                   baseOrientation=p.getQuaternionFromEuler([0, 0, np.pi/2]))
                                   
        except Exception as e:
            print(f"Errors from spawning URDF files from pybullet_data: {e}")
            raise
        
        self.obj_id = self.create_cube()
        
        # Load the robot
        orientation_quat = p.getQuaternionFromEuler([0, 0, np.pi / 2])
        try:
            self.robot_id = p.loadURDF("robotarm_revise.urdf", useFixedBase=True, baseOrientation=orientation_quat)
        except Exception as e:
            print(f"Error while loading robot URDF: {e}")
            raise
        
        # Save the number of joints of the robot
        # except for fixed camera joint and gripper joints
        self.num_tot_joints = p.getNumJoints(self.robot_id) - 3

        """
        Settings for gravity and simulation in the environment
        """
        # Set gravity and simulation timestep
        p.setGravity(0, 0, -9.81)
        self.timestep = (1. / 75. if is_for_training else 1. / 240.)  # Time interval between simulation steps
        p.setTimeStep(self.timestep)

        # Enable or disable real-time simulation
        p.setRealTimeSimulation(1 if self.realtime else 0)

        # Variables for observation and agent's goal
        # =====================Position=========================
        self.desired_place_position = np.array([0.3,0.5,0.66])
        # ======================================================
        
        self.num_actions_x = 10 # 
        self.num_actions_y = 10 # 10 * 10 image grid 
        self.center_obs_position = np.array([self.width/2, self.height/2])

        # Angle resolution: 5 degrees in radians
         # Angle variation: ±[0, 5, 10, 15, 20, 25, 30, 35, 40, 45] degrees
        self.ang_res = np.pi / 36
        self.angles = []
        # Adding positive and negative angle variations, including 0
        for i in range(-9, 10):
            self.angles.append(i * self.ang_res)
        self.num_angles = len(self.angles)

        # Define the action space and observation space
        self.action_space = spaces.Discrete(self.num_actions_x * self.num_actions_y * self.num_angles)
        self.observation_space = spaces.Box(low=0, high=2.0,
                                            shape=(self.width, self.height, 1),
                                            dtype=np.float32)  # pixel 640*480 depth image
        
        self.observation = None
        self.obj_pos = None
        # Reset the robot's pose and get camera data
        self.reset_env()
        self.rgb_img, self.depth_img = self.arm_camera()

# DRL process ===============================================
    def reset_env(self):
        """
         Resets the robot's position.
        """
        p.resetSimulation()
        self.plane_id = p.loadURDF("plane.urdf")
        self.table_id = p.loadURDF("table/table.urdf", 
                                basePosition=[0, 0, 0], 
                                baseOrientation=p.getQuaternionFromEuler([0, 0, np.pi/2]))
        
        orientation_quat = p.getQuaternionFromEuler([0, 0, np.pi / 2])
        self.robot_id = p.loadURDF("robotarm_revise.urdf", useFixedBase=True, baseOrientation=orientation_quat)

        # Use resetJointState to directly set the position of each joint
        for joint_index in range(self.num_tot_joints):
            target_position = self.joint_positions_start[joint_index]
            p.resetJointState(self.robot_id, joint_index, target_position)
        
        # Gripper
        p.resetJointState(self.robot_id,7,target_position[self.joint_positions_start[6]])
        p.resetJointState(self.robot_id,8,target_position[self.joint_positions_start[7]])
        
        # Save the new joint positions
        for joint_index in range(self.num_tot_joints):
            p.setJointMotorControl2(self.robot_id, jointIndex=joint_index,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=self.joint_positions_start[joint_index],
                                    force=500)
        p.setJointMotorControl2(self.robot_id, jointIndex= 7,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=self.joint_positions_start[6],
                                    force=500)
        p.setJointMotorControl2(self.robot_id, jointIndex= 8,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=self.joint_positions_start[7],
                                    force=500)
        
        self.obj_id = self.create_cube()
        self.rgb_img, self.depth_img = self.arm_camera()
        p.stepSimulation()

        return self.get_observation(), {}
    
    # <keep its structure>
    def step(self, action):
        """
        Executes a step of the simulation based on the provided action.

        Args:
            action (int): The action to execute: an integer encoding a pixel and orientation

        Returns:
            tuple: Observation, reward, completion flag, truncation flag, additional information.
        """
        done = False

        # Increment the number of steps
        self.num_step += 1
        target_angle = self.image2angle(action)

        # Execute the desired movement of the robot
        self.move_the_robot(self.obj_pos, target_angle)

        # Get the next observation
        observation = self.get_observation()

        # REWARD
        reward = self.calculate_reward()

        # Useful if we decide to have more steps per episode
        if self.num_step >= self.num_step_max_per_episode:
            done = True

        return observation, reward, done, False, {}
    
    # Unclear => Make it more specific
    def calculate_reward(self):
        """
        Calculates the reward.

        Can be improved by providing a more granular reward, especially if object posture needs to be imposed

        Returns:
            reward (double): positive if the object has been lifted, negative otherwise
        """
        # REWARD
        # Reward based on lifting the cube
        obj_pos, _ = p.getBasePositionAndOrientation(self.obj_id)
        distance = np.linalg.norm(obj_pos - self.desired_place_position)

        if distance < self.tolerance_reward:
            reward = 1
        else:
            reward = -2

        return reward
    
    # Unclear => Add YOLO
    def get_observation(self):
        '''
        observation : cropped depth image by YOLO
        '''
        # Initialize observation
        observation = None
        self.rgb_img, self.depth_img = self.arm_camera()

        # YOLO here
        # yolo_crop_image,target_pos = YOLO(rgb_image,depth_img)
        # observation = crop_and_resize(yolo_crop_image)
        
        target_pos = 0 # Add YOLO and erase this line
        self.obj_pos = target_pos
        self.observation = observation
        return observation
# DRL process ===============================================

# Utilities =================================================
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

    def create_cube(self):
        aabb_min, aabb_max = p.getAABB(self.table_id)
        pos_obj = [np.random.uniform(aabb_min[0]/2, aabb_max[0]/2),
                   np.random.uniform(aabb_min[1]/2, aabb_max[1]/2),
                   aabb_max[2] + 0.03]
        orientation_obj = p.getQuaternionFromEuler([0,0,0])
        
        while True:
            line_pos = [np.random.uniform(aabb_min[0], aabb_max[0]), 
                        np.random.uniform(aabb_min[1], aabb_max[1]), 
                        aabb_max[2]]
            if not np.allclose(pos_obj[:2], line_pos[:2], atol=0.1):  # 큐브와 목표 위치가 같지 않도록 설정
                break
        self.obj_id = p.loadURDF("cube_small.urdf", pos_obj, orientation_obj, useFixedBase=False)
        
        obj_aabb_min, obj_aabb_max = p.getAABB(self.obj_id)
        obj_length = (obj_aabb_max[0] - obj_aabb_min[0]) / 2
        p1 = [line_pos[0] - obj_length, line_pos[1] - obj_length, line_pos[2]]
        p2 = [line_pos[0] + obj_length, line_pos[1] - obj_length, line_pos[2]]
        p3 = [line_pos[0] + obj_length, line_pos[1] + obj_length, line_pos[2]]
        p4 = [line_pos[0] - obj_length, line_pos[1] + obj_length, line_pos[2]]

        p.addUserDebugLine(p1, p2, [1, 0, 0], 2)
        p.addUserDebugLine(p2, p3, [1, 0, 0], 2)
        p.addUserDebugLine(p3, p4, [1, 0, 0], 2)
        p.addUserDebugLine(p4, p1, [1, 0, 0], 2)
        
        return self.obj_id

    def normalize_angle(self, angle):
        """
        Normalizes an angle to the range [-pi, pi].
        Args:
            angle (double): angle to normalize
        Returns:
            normalized angle (double): Angle in the range [-pi, pi].
        """
        return np.arctan2(np.sin(angle), np.cos(angle))
    
    def crop_and_resize(self, crop_image, max_length, center_x, center_y):
        """
        Creates the final observation for the robot.
        Crops the grayscale image, resizes it, and filters it to obtain a binary image

        Returns:
            observation (np.array): Observation of the environment
        """

        half_size = max_length // 2  # Half side length of the square [pixel]

        # Calculate the crop limits, avoiding cropping beyond the edges
        start_x = max(0, center_x - half_size)
        end_x = min(self.width_img, center_x + half_size)
        start_y = max(0, center_y - half_size)
        end_y = min(self.height_img, center_y + half_size)

        # Crop the image
        cropped_image = crop_image[start_y:end_y, start_x:end_x]

        # Resize the cropped image to (currently) 37x37 pixels
        if cropped_image.size > 0:  # Avoid empty image
            resized_object = cv2.resize(cropped_image, (self.num_actions_x, self.num_actions_y),
                                        interpolation=cv2.INTER_LINEAR)
            # Binary threshold to create a black-and-white image
            _, observation = cv2.threshold(resized_object, 50, 255, cv2.THRESH_BINARY)
            observation = cv2.resize(observation, (self.obs_width, self.obs_height),
                                      interpolation=cv2.INTER_NEAREST)
            observation = observation[:, :, np.newaxis]  # Add channel (37, 37, 1)
        else:
            # White image if the crop is empty
            observation = np.full((self.obs_width, self.obs_height, 1), 255, dtype=np.uint8)

        return observation
# Utilities =================================================

# Robot move process ========================================
    # clear but check again
    def move_the_robot(self, target_position, angle, place_position):
        """
            Makes the robot perform the desired movement.
            For now, it positions itself above the object,
            1) lowers
            2) executes the grasp
            3) lifts the object

            Args:
                target_position (np.array): position for the robot to reach
                angle (double): end effector angle
        """
        # Open the gripper
        self.openGripper()

        # Position itself above the object
        new_target = np.array([target_position[0],
                               target_position[1],
                               target_position[2]])
        self.calculateIK(new_target, angle)
        self.simulate_movement()

        # Lower
        new_target = np.array([target_position[0],
                               target_position[1],
                               self.obj_size[2] * 0.65])
        self.calculateIK(new_target, angle)
        self.simulate_movement()

        # Close the gripper
        self.closeGripper()

        # Lift
        new_target = np.array([place_position[0],
                               place_position[1],
                               place_position[2]])

        # Calculate desired angles of the robot
        self.calculateIK(new_target, angle)
        # Perform simulation
        self.simulate_movement()
        # Open the gripper
        self.openGripper()

    # Unclear
    def image2angle(self,action):
        '''
        target_position : inertial target position in environment
        angle : inertial target yaw angle in environmnet
        '''
        # Decode angle
        angle = self.angles[action % self.num_angles]
        return angle
    
    # Unclear
    def calculateIK(self, target_position, angle):
        """
        Normalizes an angle to the range [-pi, pi].

        Args:
            target_position (np.array): desired position
        Returns:
            angle (double): Angle in the range [-pi, pi] for the end effector.
        """
        # Calculate, with IK, the desired angles of the joints for the desired position
        self.joint_angles_des = p.calculateInverseKinematics(self.robotId, 6, target_position,
                                                             p.getQuaternionFromEuler([0, np.pi, angle]),
                                                             maxNumIterations=50, residualThreshold=1e-3)

        # For each joint, update the control. In this case, position control
        for i in range(len(self.joint_angles_des)):
            p.setJointMotorControl2(self.robotId, i, p.POSITION_CONTROL,
                                    self.joint_angles_des[i])
    
    # Unclear
    def simulate_movement(self):
        """
        Simulates the last imposed movement to the robot.
        """

        # step: i-th iteration of the loop
        step = 0
        normalized_target_angles = [self.normalize_angle(angle) for angle in self.joint_angles_des[:7]]

        while step < self.num_simulation_steps:  # Loop until the maximum number of cycles is reached
            step += 1  # Keep track of the number of steps
            p.stepSimulation()  # Execute a simulation step

            # Likely to be removed, doing this to lighten the computational load
            if step % 10:  # Every certain number of steps, check if the point has been reached
                # Check the current angles of the joints
                current_joint_angles = [p.getJointState(self.robotId, i)[0] for i in range(7)]

                # Normalize the target and current angles to the range [-pi, pi]
                normalized_current_angles = [self.normalize_angle(angle) for angle in current_joint_angles]

                # Calculate the angular difference considering periodicity
                angle_diff = np.abs(np.array(normalized_current_angles) - np.array(normalized_target_angles))

                # If all angles are within the tolerance, end the loop
                if np.all(angle_diff < self.tolerance):
                    break
# Robot move process ========================================

# Gripper section ===========================================
    def openGripper(self, gripper_opening):
        """
        Opens the gripper

        Args:
            gripper_opening (double): how much to open the gripper
        """
        # Close the gripper (joints 9 and 10 control the panda's fingers)
        p.setJointMotorControl2(self.robot_id, 7, p.POSITION_CONTROL, np.pi/12)  # Value to adjust
        p.setJointMotorControl2(self.robot_id, 8, p.POSITION_CONTROL, -np.pi/12)

        for _ in range(10):  # 10 iterations are enough
            p.stepSimulation()
    
# Force ?
    def closeGripper(self, max_force=1.0):
        """
            Normalizes an angle to the range [-pi, pi].

            Args:
                max_force (double): force that must be felt by the gripper fingers to stop closing
        """
        # Close the gripper (joints 9 and 10 control the panda's fingers)
        p.setJointMotorControl2(self.robotId, 7, p.POSITION_CONTROL, -np.pi/12)  # Value to adjust
        p.setJointMotorControl2(self.robotId, 8, p.POSITION_CONTROL, np.pi/12)
        step = 0

        # Perform at most 100 steps for the simulation
        while step < 100:
            step += 1
            forces = []
            for joint_index in [7, 8]:
                # Read forces on joints
                joint_state = p.getJointState(self.robot_id, joint_index)
                joint_force = joint_state[3]  # The fourth element is the force
                forces.append(joint_force)

            # If one of the forces exceeds the limit, stop closing
            if any(abs(force) > max_force for force in forces):
                break

            p.stepSimulation()  # Continue simulating
# Gripper section ===========================================

    def close(self):
        """
        Closes the connection to PyBullet.
        """
        p.disconnect()
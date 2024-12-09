"""
Script for creating the environment.
Robot's goal: Raise the object to a specific height

---------------------------------------------------------------------------------------------------------------
Controllable joints of the robot:
robot_name: Joint_Type (index) ---> The angles of the revolute joints are in radians
panda_joint1: Revolute (0)
panda_joint2: Revolute (1)
panda_joint3: Revolute (2)
panda_joint4: Revolute (3)
panda_joint5: Revolute (4)
panda_joint6: Revolute (5)
panda_joint7: Revolute (6)
panda_finger_joint1: Prismatic (9)
panda_finger_joint2: Prismatic (10)

---------------------------------------------------------------------------------------------------------------
Possible actions:
[0, num_actions_x x num_actions_y x num_angles] :
Each "macro-pixel" of the observation is associated with a slight displacement of the robot relative to the central pixel coordinate.
---------------------------------------------------------------------------------------------------------------
Observation:
Binary image of dimensions n x n pixels. For now n = 37

---------------------------------------------------------------------------------------------------------------
Reward:
+1 object lifted
-2 object not lifted

Possible modification: reward as a function of the difference from the desired position
---------------------------------------------------------------------------------------------------------------

TODO:
 - Try inserting elements of different shapes
 - Increase the number of angles
 - Consider the posture of the object at the final instant
"""

"""
    LIBRARIES
"""
import random
import gymnasium as gym
import pybullet as p
import pybullet_data
import numpy as np
import cv2
from gymnasium import spaces
import time


class RobotEnv(gym.Env):
    """
    Class representing the environment for training the robot using PyBullet.
    The environment supports both GUI and non-GUI mode and can be configured for real-time use or not.
    """

    def __init__(self, is_for_training=False):
        """
        Initializes the robot's environment.

        Args:
            is_for_training (bool): If True, uses PyBullet's graphical interface and
                                    the simulation is run in real-time.
        """
        super(RobotEnv, self).__init__()

        """
        ---------------------------------------------------------------------------------------------------------------
        VARIABLES
        ---------------------------------------------------------------------------------------------------------------
        """
        # ------------------------------------------------------------------------------------------------------------

        # Variables that regulate the various aspects of the training and simulation environment
        self.use_gui = not is_for_training
        self.realtime = not is_for_training

        # Variables for the environment
        self.num_step_max_per_episode = 1  # One step per episode --> try to grasp and repeat the cycle
        self.num_step = 0  # Number of steps taken in the episode
        self.num_simulation_steps = 2000  # Maximum number of simulation steps for the robot's movement
        self.tolerance = 0.005  # Tolerance value for the robot's movement
        self.tolerance_reward = 0.05

        # --------------------------------------------OBJECTS-------------------------------------------------------

        # Variables for the objects
        self.obj_size = np.array([0.02, 0.05, 0.02])  # Dimensions
        self.obj_x_min = -0.15  # Maximum and minimum coordinates
        self.obj_x_max = 0.15
        self.obj_y_min = 0.40
        self.obj_y_max = 0.70
        # Commented out because, for now, the end effector and objects can assume the same orientations
        # self.obj_yaw = [0, -np.pi / 4, -np.pi / 2, np.pi / 4]  # Object posture (r,p,y)
        self.obj_pitch = 0
        self.obj_roll = 0

        # ------------------------------------------CAMERA------------------------------------------------------------

        # Coordinates pointed by the cameras (in the center of the possible object positions)
        camera_point_to_x = np.mean([self.obj_x_min, self.obj_x_max])
        camera_point_to_y = np.mean([self.obj_y_min, self.obj_y_max])
        self.camera_point_to = [camera_point_to_x, camera_point_to_y, 0]

        # Dimensions of captured image
        self.width_img = 401
        self.height_img = 401

        # Camera field of view (FOV) for observations in radians
        field_of_view = 60
        fov_rad = np.radians(field_of_view)

        # Camera distance for observations from the pointed coordinates
        self.camera_distance = 0.4

        # --------------------------------------------ROBOT-----------------------------------------------------------

        # Robot

        # Initial positions of robot joints
        self.joint_positions_start = [0.0, 0.0, 0.0, -np.pi / 3, 0.0, np.pi / 1.75, np.pi / 4, 0.0, 0.0, 0.03, 0.03,
                                      0.0]
        self.joint_angles_des = None
        self.joint_velocities = np.array([0.1, 0.1, 0.1, 0.1, 0.20, 0.25, 0.55, 0.0, 0.0, 0.0, 0.0]) * 2

        # self.angles = [0, -np.pi / 4, -np.pi / 2, np.pi / 4]
        self.angles = [0, -np.pi / 6, -np.pi / 3, -np.pi / 2,  2 * np.pi / 6, np.pi / 6]
        self.num_angles = len(self.angles)
        self.obj_yaw = self.angles  # Object posture (r,p,y)

        # --------------------------------------------ACTIONS AND OBSERVATIONS--------------------------------------------

        # Variables for observation and agent's goal
        self.desired_cube_height = 0.50
        self.obs_width = 37
        self.obs_height = 37
        self.num_actions_x = 7
        self.num_actions_y = 7
        self.center_obs_position = np.array([0, 0])
        self.delta_l = 0  # Unit movement for the robot

        # Calculate half the real-world view width
        self.semi_width = self.camera_distance * np.tan(fov_rad / 2)

        # Minimum and maximum coordinates for (x, y), pointed by the observation camera
        self.world_x_min = self.camera_point_to[0] - self.semi_width
        self.world_x_max = self.camera_point_to[0] + self.semi_width
        self.world_y_min = self.camera_point_to[1] - self.semi_width
        self.world_y_max = self.camera_point_to[1] + self.semi_width

        # ---------------------------------------------------ENVIRONMENT CREATION--------------------------------------

        """
        Connect to PyBullet
        """
        try:
            if self.use_gui:
                p.connect(p.GUI)
                p.resetDebugVisualizerCamera(
                    cameraDistance=1.6,
                    cameraYaw=120,
                    cameraPitch=-50,
                    cameraTargetPosition=self.camera_point_to
                )
            else:
                p.connect(p.DIRECT)
        except Exception as e:
            print(f"Error during PyBullet connection: {e}")
            raise

        """
        Camera from which observations are obtained
        """
        # Set up the camera for viewing
        self.view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=self.camera_point_to,
            distance=self.camera_distance,  # Camera distance
            yaw=0,
            pitch=-90,
            roll=0,
            upAxisIndex=2  # Z-axis is considered the "up" axis
        )

        self.proj_matrix = p.computeProjectionMatrixFOV(
            fov=field_of_view,  # Camera field of view
            aspect=float(self.width_img) / self.height_img,  # Aspect ratio
            nearVal=0.1,  # Near clipping distance
            farVal=5.0  # Far clipping distance
        )

        """
        Load the elements present in the environment:
            - plane
            - robot
            - objects
        """
        # Set the path for URDF files
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Load the plane
        try:
            self.planeId = p.loadURDF("plane.urdf")
        except Exception as e:
            print(f"Error while loading plane URDF: {e}")
            raise

        # Definition of available objects with a dictionary mapping indices to creation methods
        available_objects = {
            0: self.create_l_object,
            1: self.create_parallelepiped,
            2: self.create_z_object
        }

        # Generate object position and orientation
        pos_obj, orientation_obj = self.generate_new_pose()

        # Random generation or controlled selection of object type
        # obj_type = random.randint(0, len(available_objects) - 1)  # Randomly select one of the objects
        obj_type = 0

        # Check if the selected object is valid
        try:
            self.objId = available_objects[obj_type](pos_obj, orientation_obj)
        except KeyError:
            print(f"Error: selected object type ({obj_type}) is not valid. Please select a valid type.")
            raise

        # Load the robot
        orientation_quat = p.getQuaternionFromEuler([0, 0, np.pi / 2])
        try:
            self.robotId = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True, baseOrientation=orientation_quat)
        except Exception as e:
            print(f"Error while loading robot URDF: {e}")
            raise

        # Save the number of joints of the robot
        self.num_tot_joints = p.getNumJoints(self.robotId)  # There are joints that are not controllable

        # Reset the robot's pose
        self.reset_robot()

        """
        Settings for gravity and simulation in the environment
        """
        # Set gravity and simulation timestep
        p.setGravity(0, 0, -9.81)
        self.timestep = (1. / 75. if is_for_training else 1. / 240.)  # Time interval between simulation steps
        p.setTimeStep(self.timestep)

        # Enable or disable real-time simulation
        p.setRealTimeSimulation(1 if self.realtime else 0)

        """
        Action and observation space
        """
        # Define the action space and observation space
        self.action_space = spaces.Discrete(self.num_actions_x * self.num_actions_y * self.num_angles)

        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(self.obs_width, self.obs_height, 1),
                                            dtype=np.uint8)  # Binary 37*37 image

    def reset(self, seed=None, options=None):
        """
        Resets the environment at the beginning of a new episode.

        Args:
            seed (int, optional): Seed for random cube pose generation
                                    In future implementations, it may also influence the number
                                    and size of objects.
                                    Default is None.
            options (dict, optional): Options for reset. Default is None.

        Returns:
            observation (np.array): Environment observation.
            dict: Additional information (empty for now but may come in handy).
        """

        if seed is not None:
            np.random.seed(seed)  # Set seed for reproducibility

        # Reset step count
        self.num_step = 0

        # Reset cube pose
        pos_obj, orientation_obj = self.generate_new_pose()
        p.resetBasePositionAndOrientation(self.objId, pos_obj, orientation_obj)

        # Reset robot pose
        self.reset_robot()

        return self.get_observation(), {}

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

        target_position, angle = self.action_to_position_and_observation(action)

        # Execute the desired movement of the robot
        self.move_the_robot(target_position, angle)

        # Get the next observation
        observation = self.get_observation()

        # REWARD
        reward = self.calculate_reward()

        # Useful if we decide to have more steps per episode
        if self.num_step >= self.num_step_max_per_episode:
            done = True

        return observation, reward, done, False, {}

    def close(self):
        """
        Closes the connection to PyBullet.
        """
        p.disconnect()

    def calculate_reward(self):
        """
        Calculates the reward.

        Can be improved by providing a more granular reward, especially if object posture needs to be imposed

        Returns:
            reward (double): positive if the object has been lifted, negative otherwise
        """
        # REWARD
        # Reward based on lifting the cube
        obj_pos, _ = p.getBasePositionAndOrientation(self.objId)
        distance = np.linalg.norm(obj_pos[2] - self.desired_cube_height)

        if distance < self.tolerance_reward:
            reward = 1
        else:
            reward = -2

        return reward

    def action_to_position_and_observation(self, action):
        """
        Translates the action into a position and an orientation
        Example: 4 possible orientations
        action = 0,1,2,3,4,5,6,7,8,9,...
            [0,1,2,3] -> angles α, β, γ, δ associated with the 0th macro-pixel of the observation
            [4,5,6,7] -> angles α, β, γ, δ associated with the 1st macro-pixel of the observation
            ...

        Example: 3 macro-pixels per dimension in the image, each macro-pixel is assigned an identifier:
            0 1 2
            3 4 5
            6 7 8
        Thanks to these identifiers, the position of the gripper can be corrected

        Args:
            action (int): The action to execute: an integer provided by the trained model

        Returns:
            target_position: Position that the robot must reach for an efficient grasp
            target_orientation: Orientation that the robot must reach for an efficient grasp
        """

        # Decode angle
        angle = self.angles[action % self.num_angles]

        # Decode pixel
        # action = 0,1,2,3,4,5,6,7,8,9,... --> 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, ...
        action = action // self.num_angles

        # Map the pixel identifier with the pixel coordinates of the observation
        pointed_pixel = [action // self.num_actions_x, action % self.num_actions_x]

        # Calculate world coordinates to provide to the robot for inverse kinematics
        target_position_x = self.center_obs_position[0] + (pointed_pixel[0] - (self.num_actions_x // 2)) * self.delta_l
        target_position_y = self.center_obs_position[1] + (self.num_actions_y // 2 - pointed_pixel[1]) * self.delta_l
        target_position = np.array([target_position_x, target_position_y])

        return target_position, angle

    def get_observation(self):
        """
        Gets the current environment observation in the form of an image and calculates the coordinates
        of the central pixel
        Returns:
            observation (np.array): Binary image representing obstacle (black) or no obstacle (white).
        """

        # Initialize observation
        observation = None

        # Capture the image
        rgb_image, gray_image = self.capture_image()

        # Detect contours
        edge, contours = self.detect_contours(gray_image)

        # Based on contours, calculate delta_l and the central pixel position in world coordinates
        if contours:
            for contour in contours:
                # Find the top-left pixel and dimensions of the bounding box
                x, y, w, h = cv2.boundingRect(contour)

                # Central pixel
                center_x, center_y = x + w // 2, y + h // 2

                # The goal is to obtain a square observation:
                # Find the maximum size of the bounding box, which will represent the side of the square
                max_length = max(w, h)  # Side length of the square [pixel]

                # Calculate the real-world coordinates from the central pixel of the bounding box
                self.center_obs_position = self.pixel_to_world(center_x, center_y)

                # Since "w" and "h" are not constant, it is necessary to calculate the step size for
                # moving between adjacent actions (for now delta_l is the same for x and y)
                self.delta_l = (2 * self.semi_width * max_length / self.width_img) / self.num_actions_x  # [world]

                observation = self.crop_and_resize(gray_image, max_length, center_x, center_y)

        else:
            # If no object is found, create an entirely white image
            observation = np.full((self.obs_width, self.obs_height, 1), 255, dtype=np.uint8)

        return observation

    def capture_image(self):
        """
        Captures the RGB image and processes it.
        Note: OpenCV works with BGR, so conversions are necessary (probably need to recheck)

        Returns:
            rgb_image (np.array): RGB image captured by the camera
            gray_image (np.array): Grayscale image captured by the camera
        """
        _, _, rgb_image, _, _ = p.getCameraImage(width=self.width_img, height=self.height_img,
                                                 viewMatrix=self.view_matrix,
                                                 projectionMatrix=self.proj_matrix)

        # Image processing with OpenCV to find contours
        rgb_image = np.array(rgb_image, dtype=np.uint8)[:, :, :3]  # In case, remove alpha channel
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)

        return rgb_image, gray_image

    def detect_contours(self, gray_image):
        """
        Detects the contours of the object present in the image.
        For now, the bounding box is rectangular and not tilted

        Returns:
            edges (np.array): Image with the detected edges representing the object contours
            contours (list): List of detected contours. Each contour is a sequence of points
                             defining the perimeter of the detected object in the image.
        """
        # Image with the edges of the detected object
        edges = cv2.Canny(cv2.GaussianBlur(gray_image, (5, 5), 0), 150, 200)

        # Find the contour and the middle point of the bounding box
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        return edges, contours

    def crop_and_resize(self, gray_image, max_length, center_x, center_y):
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
        cropped_image = gray_image[start_y:end_y, start_x:end_x]

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

    def pixel_to_world(self, pixel_x, pixel_y):
        """
        Converts the coordinates of a pixel captured by the camera to real-world coordinates.

        Args:
            pixel_x (int): x coordinate of the pixel in the image (origin at top left).
            pixel_y (int): y coordinate of the pixel in the image.

        Returns:
            [world_x, world_y] (list): Coordinates (x, y) in the real world corresponding to the provided pixel.
        """
        # Map the pixel coordinates to world coordinates
        world_x = np.interp(pixel_x, [0, self.width_img - 1], [self.world_x_min, self.world_x_max])
        world_y = np.interp(pixel_y, [0, self.height_img - 1], [self.world_y_max, self.world_y_min])

        return [world_x, world_y]

    def generate_new_pose(self):
        """
         Generates a new position and orientation of the object

         Returns:
             pos_obj (list): new position of the object
             orientation_obj (double): new orientation
         """
        # Random generation of new cube position and orientation
        pos_obj = [np.random.uniform(low=self.obj_x_min, high=self.obj_x_max),
                   np.random.uniform(low=self.obj_y_min, high=self.obj_y_max),
                   self.obj_size[2]]

        yaw = random.choice(self.obj_yaw)
        orientation_obj = p.getQuaternionFromEuler([self.obj_roll,
                                                     self.obj_pitch,
                                                     yaw])

        return pos_obj, orientation_obj

    def reset_robot(self):
        """
         Resets the robot's position.
        """
        # Use resetJointState to directly set the position of each joint
        for joint_index in range(self.num_tot_joints):
            target_position = self.joint_positions_start[joint_index]
            p.resetJointState(self.robotId, joint_index, target_position)

        # Save the new joint positions
        for joint_index in range(self.num_tot_joints):
            p.setJointMotorControl2(self.robotId, jointIndex=joint_index,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=self.joint_positions_start[joint_index],
                                    force=500)

        p.stepSimulation()

    def normalize_angle(self, angle):
        """
        Normalizes an angle to the range [-pi, pi].
        Args:
            angle (double): angle to normalize
        Returns:
            normalized angle (double): Angle in the range [-pi, pi].
        """
        return np.arctan2(np.sin(angle), np.cos(angle))

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

    def calculateIK(self, target_position, angle):
        """
        Normalizes an angle to the range [-pi, pi].

        Args:
            target_position (np.array): desired position
        Returns:
            angle (double): Angle in the range [-pi, pi] for the end effector.
        """
        # Calculate, with IK, the desired angles of the joints for the desired position
        self.joint_angles_des = p.calculateInverseKinematics(self.robotId, 11, target_position,
                                                             p.getQuaternionFromEuler([0, np.pi, angle]),
                                                             maxNumIterations=50, residualThreshold=1e-3)

        # For each joint, update the control. In this case, position control
        for i in range(len(self.joint_angles_des)):
            p.setJointMotorControl2(self.robotId, i, p.POSITION_CONTROL,
                                    self.joint_angles_des[i], maxVelocity=self.joint_velocities[i])

    def openGripper(self, gripper_opening):
        """
        Opens the gripper

        Args:
            gripper_opening (double): how much to open the gripper
        """
        # Close the gripper (joints 9 and 10 control the panda's fingers)
        p.setJointMotorControl2(self.robotId, 9, p.POSITION_CONTROL, gripper_opening)  # Value to adjust
        p.setJointMotorControl2(self.robotId, 10, p.POSITION_CONTROL, gripper_opening)

        for _ in range(10):  # 10 iterations are enough
            p.stepSimulation()

    def closeGripper(self, max_force=1.0):
        """
            Normalizes an angle to the range [-pi, pi].

            Args:
                max_force (double): force that must be felt by the gripper fingers to stop closing
        """
        # Close the gripper (joints 9 and 10 control the panda's fingers)
        p.setJointMotorControl2(self.robotId, 9, p.POSITION_CONTROL, 0.01)  # Value to adjust
        p.setJointMotorControl2(self.robotId, 10, p.POSITION_CONTROL, 0.01)
        step = 0

        # Perform at most 100 steps for the simulation
        while step < 100:
            step += 1
            forces = []
            for joint_index in [9, 10]:
                # Read forces on joints
                joint_state = p.getJointState(self.robotId, joint_index)
                joint_force = joint_state[3]  # The fourth element is the force
                forces.append(joint_force)

            # If one of the forces exceeds the limit, stop closing
            if any(abs(force) > max_force for force in forces):
                break

            p.stepSimulation()  # Continue simulating

    def move_the_robot(self, target_position, angle):
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

        # Position itself above the object
        new_target = np.array([target_position[0],
                               target_position[1],
                               self.obj_size[2] * 4])
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
        new_target = np.array([target_position[0],
                               target_position[1],
                               self.desired_cube_height])

        # Calculate desired angles of the robot
        self.calculateIK(new_target, angle)
        # Perform simulation
        self.simulate_movement()

    def create_l_object(self, position, orientation):
        """
        Creates an L-shaped object.

        Args:
            position (np.array): new position of the object
            orientation (double): new orientation of the object
        Returns:
            l_shape_id: ID of the created object
        """

        # Define dimensions for the blocks that will form the L shape
        block1_size = self.obj_size  # Dimensions of the first block
        block2_size = [block1_size[1], block1_size[0], block1_size[2]]  # Dimensions of the second block

        # Define the base position of the L-shaped object
        base_position = [0, 0, 0]

        # Define the relative positions of the blocks
        block1_position = [0, 0, 0]  # Offset of the first block
        block2_position = [block1_size[0] - block1_size[1], block1_size[1], 0]  # Offset of the second block

        # Create the "collision shapes" for the two blocks
        block1_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=block1_size)
        block2_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=block2_size)

        # Create the "visual shapes" for the two blocks (optional, for visualization)
        block1_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=block1_size, rgbaColor=[0, 0, 0.5, 1])
        block2_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=block2_size, rgbaColor=[0, 0, 0.5, 1])

        # Define the mass of the object
        mass = 1

        # Create the L-shaped object using two blocks
        l_shape_id = p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=block1_collision,
            baseVisualShapeIndex=block1_visual,
            basePosition=np.add(base_position, block1_position),
            # Add the second block as a "link" to the first
            linkMasses=[mass],
            linkCollisionShapeIndices=[block2_collision],
            linkVisualShapeIndices=[block2_visual],
            linkPositions=[block2_position],
            linkOrientations=[p.getQuaternionFromEuler([0, 0, 0])],
            linkInertialFramePositions=[[0, 0, 0]],
            linkInertialFrameOrientations=[[0, 0, 0, 1]],
            linkParentIndices=[0],
            linkJointTypes=[p.JOINT_FIXED],  # Fixed joint to keep the blocks together
            linkJointAxis=[[0, 0, 0]]
        )

        # Reset the position and orientation of the object
        p.resetBasePositionAndOrientation(l_shape_id, position, orientation)

        return l_shape_id

    def create_parallelepiped(self, position, orientation):
        """
        Creates a parallelepiped object.

        Args:
            position (np.array): new position of the object
            orientation (double): new orientation of the object
        Returns:
            cubeId: ID of the created object
        """
        cubeId = p.createMultiBody(
            baseMass=1,
            baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=self.obj_size),
            baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=self.obj_size, rgbaColor=[0, 0, 0.5, 1]),
            basePosition=position,
            baseOrientation=orientation
        )
        return cubeId

    def create_z_object(self, position, orientation):
        """
        Creates a Z-shaped object.

        Args:
            position (np.array): new position of the object
            orientation (double): new orientation of the object
        Returns:
            z_shape_id: ID of the created object
        """
        # Define dimensions for the blocks that will form the Z shape
        block1_size = self.obj_size * 0.6  # Dimensions of the first block
        block1_size[2] = self.obj_size[2]
        block1_size[0] = block1_size[0] * 0.8  # Dimensions of the first block
        block2_size = [block1_size[1] * 1.33, block1_size[0], block1_size[2]]  # Dimensions of the second block (diagonal)
        block3_size = block1_size  # Dimensions of the third block (same as the first)

        # Define the base position of the Z-shaped object
        base_position = [0, 0, 0]

        # Define the relative positions of the blocks
        block1_position = [0, 0, 0]  # Top block
        block3_position = [-2.625 * block1_size[1], 0, 0]  # Bottom block
        block2_position = [block3_position[0] / 2, 0, 0]  # Middle block (diagonal)

        # Create the "collision shapes" for the three blocks
        block1_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=block1_size)
        block2_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=block2_size)
        block3_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=block3_size)

        # Create the "visual shapes" for the three blocks (optional, for visualization)
        block1_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=block1_size, rgbaColor=[0, 0, 0.5, 1])
        block2_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=block2_size, rgbaColor=[0, 0, 0.5, 1])
        block3_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=block3_size, rgbaColor=[0, 0, 0.5, 1])

        # Define the mass of the object
        mass = 1

        # Create the Z-shaped object using three blocks
        z_shape_id = p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=block1_collision,
            baseVisualShapeIndex=block1_visual,
            basePosition=np.add(base_position, block1_position),
            # Add the second and third blocks as "links" to the first
            linkMasses=[mass, mass],
            linkCollisionShapeIndices=[block2_collision, block3_collision],
            linkVisualShapeIndices=[block2_visual, block3_visual],
            linkPositions=[block2_position, block3_position],
            linkOrientations=[p.getQuaternionFromEuler([0, 0, np.pi / 6]), p.getQuaternionFromEuler([0, 0, 0])],
            linkInertialFramePositions=[[0, 0, 0], [0, 0, 0]],
            linkInertialFrameOrientations=[[0, 0, 0, 1], [0, 0, 0, 1]],
            linkParentIndices=[0, 0],
            linkJointTypes=[p.JOINT_FIXED, p.JOINT_FIXED],  # Fixed joints to keep the blocks together
            linkJointAxis=[[0, 0, 0], [0, 0, 0]]
        )

        # This is done because otherwise more complex shapes end up outside the camera frame
        position[0] = position[0] * 0.35
        position[1] = position[1] * 0.35

        # Reset the position and orientation of the object
        p.resetBasePositionAndOrientation(z_shape_id, position, orientation)

        return z_shape_id

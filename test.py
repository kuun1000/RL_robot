import pybullet as p
import pybullet_data
import time
import numpy as np
# Connect to PyBullet
physicsClient = p.connect(p.GUI)

# Set additional search path to find PyBullet's URDF files
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Load plane for ground reference
planeId = p.loadURDF("plane.urdf")

# Load the table URDF model provided by PyBullet
table_position = [0, 0, 0]
tableId = p.loadURDF("table/table.urdf", 
                                   basePosition=[0, 0, 0], 
                                   baseOrientation=p.getQuaternionFromEuler([0, 0, np.pi/2]))

# Set gravity in the simulation
p.setGravity(0, 0, -9.8)

# Load the robot arm (e.g., KUKA iiwa) on top of the table
# Adjust the position so that the arm is on top of the table
robot_arm_start_position = [-0.5, 0, 0.67]
robot_arm_start_orientation = [0, 0, -np.pi/2]
iiwaId = p.loadURDF("robotarm_revise.urdf",
                    basePosition=robot_arm_start_position,
                    baseOrientation=p.getQuaternionFromEuler(robot_arm_start_orientation),
                    useFixedBase=True)

def openGripper():
        """
        Opens the gripper

        Args:
            gripper_opening (double): how much to open the gripper
        """
        # Close the gripper (joints 9 and 10 control the panda's fingers)
        p.setJointMotorControl2(iiwaId, 7, p.POSITION_CONTROL, -np.pi/12)  # Value to adjust
        p.setJointMotorControl2(iiwaId, 8, p.POSITION_CONTROL, np.pi/12)

        for _ in range(100):  # 10 iterations are enough
            p.stepSimulation()

# Run the simulation for some time
for i in range(1000):
    p.stepSimulation()
    openGripper()
    time.sleep(1./240.)  # Adding a delay for better visualization
    

# Disconnect from PyBullet
p.disconnect()

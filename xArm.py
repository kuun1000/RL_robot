from typing import Optional

import numpy as np
from gymnasium import spaces

from panda_gym.envs.core import PyBulletRobot
from panda_gym.pybullet import PyBullet


class xArm(PyBulletRobot):
    """xArm robot in PyBullet.
    
    Args:
        sim (PyBullet): Simulation instance
        block_gripper (bool, optional): Whether the gripper is blocked. Defaults to False.
        base_position (np.ndarray, optional): Position of the base of the robot, as (x, y, z). Defaults to (0, 0, 0).
        control_type (str, optional): "ee" to control end-effector displacement or "joints" to control joint angles. Defaults to "ee".
    """

    def __init__(
            self,
            sim: PyBullet,
            block_gripper: bool = False,
            base_position: Optional[np.ndarray] = None,
            control_type: str = "ee",
    ) -> None:
        base_position = base_position if base_position is not None else np.array([-0.5, 0, 0])
        self.block_gripper = block_gripper
        self.control_type = control_type
        n_action = 3 if self.control_type == "ee" else 6    # control (x, y, z) if "ee", else control the 6 joints
        n_action += 0 if self.block_gripper else 1
        action_space = spaces.Box(-1.0, 1.0, shape=(n_action, ), dtype=np.float32)
        super().__init__(
            sim,
            body_name="xarm",
            file_name="lite_6_robotarm.urdf",
            base_position=base_position,
            action_space=action_space,
            joint_indices=np.array([0, 1, 2, 3, 4, 5, 7, 8]),
            joint_forces=np.array([87.0, 87.0, 87.0, 12.0, 120.0, 120.0, 170.0, 170.0]),

        )

        self.fingers_indices = np.array([7, 8])
        self.neutral_joint_values = np.array([0.00, -0.52, 0.00, -1.57, 1.57, 0.00, 0.00])
        self.ee_link = 6
        self.sim.set_lateral_friction(self.body_name, self.fingers_indices[0], lateral_friction=1.0)
        self.sim.set_lateral_friction(self.body_name, self.fingers_indices[1], lateral_friction=1.0)
        self.sim.set_spinning_friction(self.body_name, self.fingers_indices[0], spinning_friction=0.001)
        self.sim.set_spinning_friction(self.body_name, self.fingers_indices[1], spinning_friction=0.001)

    def set_action(self, action: np.ndarray) -> None:
        action = action.copy()  # ensure action don't change
        action = np.clip(action, self.action_space.low, self.action_space.high)
        if self.control_type == "ee":
            # 엔드 이펙터의 변위를 제어하는 경우
            ee_displacement = action[:3]
            target_arm_angles = self.ee_displacement_to_target_arm_angles(ee_displacement)
        else:
            # 관절 각도를 제어하는 경우
            arm_joint_ctrl = action[:6]  # 주 관절 6개에 대한 제어 값
            target_arm_angles = self.arm_joint_ctrl_to_target_arm_angles(arm_joint_ctrl)

        if self.block_gripper:
            target_fingers_width = 0.0
            target_angles = np.concatenate((target_arm_angles, [target_fingers_width / 2, target_fingers_width / 2]))
        else:
            fingers_ctrl = action[-1] * 0.2  # limit maximum change in position
            fingers_width = self.get_fingers_width()
            target_fingers_width = fingers_width + fingers_ctrl
            target_angles = np.concatenate((target_arm_angles, [target_fingers_width / 2, target_fingers_width / 2]))

        # target_angles의 길이가 joint_indices와 일치하는지 확인
        assert len(target_angles) == len(self.joint_indices), (
            f"Number of target angles {len(target_angles)} does not match "
            f"the number of joint indices {len(self.joint_indices)}"
        )
        
        self.control_joints(target_angles=target_angles)

    def ee_displacement_to_target_arm_angles(self, ee_displacement: np.ndarray) -> np.ndarray:
        """Compute the target arm angles from the end-effector displacement
        
        Args:
            ee_displacement (np.ndarray): End-effector displacement, as (dx, dy, dz).

        Returns:
            np.ndarray: Target arm angles, as the angles of the 6 arm joints
        """
        ee_displacement = ee_displacement[:3] * 0.05    # limit maximum change in position
        ee_position = self.get_ee_position()
        target_ee_position = ee_position + ee_displacement
        # Clip the height target. For some reason, it has a great impact on learning
        target_ee_position[2] = np.max((0, target_ee_position[2]))
        # Compute the new joint angles
        target_arm_angles = self.inverse_kinematics(
            link=self.ee_link, position=target_ee_position, orientation=np.array([1.0, 0.0, 0.0, 0.0])
        )
        target_arm_angles = target_arm_angles[:6]
        return target_arm_angles
    
    def arm_joint_ctrl_to_target_arm_angles(self, arm_joint_ctrl: np.ndarray) -> np.ndarray:
        """Compute the target arm angles from the arm joint control.
        
        Args:
            arm_joint_ctrl (np.ndarray): Control of the 5 joints.

        Returns:
            np.ndarray: Target arm angles, as the angles of the 5 arm joints.
        """
        arm_joint_ctrl = arm_joint_ctrl * 0.05  # limit maximum change in position
        # Get the current position and the target position
        current_arm_joint_angles = np.array([self.get_joint_angle(joint=i) for i in range(6)])
        target_arm_angles = current_arm_joint_angles + arm_joint_ctrl
        return target_arm_angles
    
    def get_obs(self) -> np.ndarray:
        # End-effector position and velocity
        ee_position = np.array(self.get_ee_position())
        ee_velocity = np.array(self.get_ee_velocity())
        # Fingers opening
        if not self.block_gripper:
            fingers_width = self.get_fingers_width()
            observation = np.concatenate((ee_position, ee_velocity, [fingers_width]))
        else:
            observation = np.concatenate((ee_position, ee_velocity))
        return observation
    
    def reset(self) -> None:
        self.set_joint_neutral()

    def set_joint_neutral(self) -> None:
        """Set the robot to its neutral pose."""
        self.set_joint_angles(self.neutral_joint_values)

    def get_fingers_width(self) -> float:
        """Get the distance between the fingers"""
        finger1 = self.sim.get_joint_angle(self.body_name, self.fingers_indices[0])
        finger2 = self.sim.get_joint_angle(self.body_name, self.fingers_indices[1])
        return finger1 + finger2
    
    def get_ee_position(self) -> np.ndarray:
        """Returns the position of the end-effector as (x, y, z)"""
        return self.get_link_position(self.ee_link)
    
    def get_ee_velocity(self) -> np.ndarray:
        """Returns the velocity of the end-effector as (vx, vy, vz)"""
        return self.get_link_velocity(self.ee_link)
    


# 테스트
# from panda_gym.pybullet import PyBullet

# sim = PyBullet(render_mode="human")
# robot = xArm(sim)

# for _ in range(10000):
#     robot.set_action(np.array([1.0]))
#     sim.step()
from panda_gym.envs.core import RobotTaskEnv
from panda_gym.pybullet import PyBullet
from custom_robot import CustomRobot
from custom_task import PickAndPlace

class MyRobotTaskEnv(RobotTaskEnv):
    """My robot-task environment."""

    def __init__(self, render_mode):
        sim = PyBullet(render_mode=render_mode)
        robot = CustomRobot(sim)
        task = PickAndPlace(sim)
        super().__init__(robot, task)
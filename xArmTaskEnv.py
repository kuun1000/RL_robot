from panda_gym.envs.core import RobotTaskEnv
from panda_gym.pybullet import PyBullet
from xArm import xArm
from pick_and_place import PickAndPlace

class MyRobotTaskEnv(RobotTaskEnv):
    """My robot-task environment."""

    def __init__(self, render_mode='human'):
        sim = PyBullet(render_mode=render_mode)
        robot = xArm(sim)
        task = PickAndPlace(sim)
        super().__init__(robot, task)


# 테스트
# env = MyRobotTaskEnv(render_mode="human")

# observation, info = env.reset()

# for _ in range(1000):
#     action = env.action_space.sample() # random action
#     observation, reward, terminated, truncated, info = env.step(action)

#     if terminated or truncated:
#         observation, info = env.reset()
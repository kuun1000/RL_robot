import cv2
from stable_baselines3.common.env_checker import check_env
from XarmRobot import XarmRobotEnv
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
import warnings

def create_test_env():
    env = XarmRobotEnv(is_for_training=False)
    return Monitor(env)  # Avvolgi l'ambiente con Monitor

def create_training_env():
    env = XarmRobotEnv(is_for_training=True)
    return Monitor(env)  # Avvolgi l'ambiente con Monitor


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    model = DQN.load("dqn_Z_6.zip")

    # Sopprime il FutureWarning
    warnings.simplefilter(action='ignore', category=FutureWarning)

    # test_env = create_no_gui_env()
    test_env = create_test_env()

    # Controlla se l'ambiente rispetta i canoni di StableBaseline3
    # check_env(test_env, warn=True)

    obs, _ = test_env.reset()

    # Test del modello addestrato
    for i in range(100):
        action, _states = model.predict(obs, deterministic=True)
        # action = test_env.action_space.sample()
        _, reward, _, _, _ = test_env.step(action)

        obs, _ = test_env.reset()

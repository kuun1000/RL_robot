import gymnasium as gym
import panda_gym
from stable_baselines3 import PPO

# Create gym training environment
env = gym.make(
    "PandaPickAndPlace-v3",
    render_mode="human",
    reward_type="dense",  # "dense" or "sparse"
    control_type="ee",  # "ee" or "joints"
)

# Set up PPO model
PPO_model = PPO(
    "MultiInputPolicy",
    env,
    verbose=1,
    tensorboard_log="./runs",
    learning_rate=0.001,
)


# Train agent
PPO_model.learn(total_timesteps=200000)

# Save trained model
PPO_model.save("PandaReach_v3_PPO_model")
import torch
from Xarm_env import XArmEnv

def main():
    # Create an instance of the XArm environment
    env = XArmEnv(steps=500, train=True)

    # Set the parameters for training
    episodes = 100
    batch_size = 64
    gamma = 0.99
    polyak = 0.995
    policy_noise = 0.2
    noise_clip = 0.5
    policy_delay = 2
    exploration_noise = 0.1
    # Train the agent using the specified parameters
    env.train_model(
        episodes=episodes,
        batch_size=batch_size,
        gamma=gamma,
        polyak=polyak,
        policy_noise=policy_noise,
        noise_clip=noise_clip,
        policy_delay=policy_delay,
        exploration_noise = exploration_noise
    )

if __name__ == "__main__":
    main()
import gym
from gym import spaces
import numpy as np

class Grid3DEnv(gym.Env):
    def __init__(self, grid_size=(10, 10, 10)):
        super(Grid3DEnv, self).__init__()

        # Define the 3D grid space
        self.grid_size = grid_size
        self.state = np.zeros(grid_size)
        
        # Observation space: The agent's current position
        self.observation_space = spaces.Box(low=0, high=max(grid_size)-1, shape=(3,), dtype=np.int32)

        # Action space: The possible moves in 3D space (up, down, left, right, forward, backward)
        self.action_space = spaces.Discrete(6)  # 6 actions for 3D movement
        
        # Define starting position
        self.agent_pos = np.array([0, 0, 0])

        # Define goal position (for example, the farthest corner)
        self.goal_pos = np.array([grid_size[0]-1, grid_size[1]-1, grid_size[2]-1])

    def reset(self):
        # Reset agent position to the start
        self.agent_pos = np.array([0, 0, 0])
        return self.agent_pos

    def step(self, action):
        # Define how the agent moves in 3D space
        if action == 0 and self.agent_pos[0] < self.grid_size[0] - 1:  # move right
            self.agent_pos[0] += 1
        elif action == 1 and self.agent_pos[0] > 0:  # move left
            self.agent_pos[0] -= 1
        elif action == 2 and self.agent_pos[1] < self.grid_size[1] - 1:  # move forward
            self.agent_pos[1] += 1
        elif action == 3 and self.agent_pos[1] > 0:  # move backward
            self.agent_pos[1] -= 1
        elif action == 4 and self.agent_pos[2] < self.grid_size[2] - 1:  # move up
            self.agent_pos[2] += 1
        elif action == 5 and self.agent_pos[2] > 0:  # move down
            self.agent_pos[2] -= 1

        # Calculate reward and check if goal is reached
        if np.array_equal(self.agent_pos, self.goal_pos):
            reward = 1  # Reward for reaching the goal
            done = True
        else:
            reward = -0.1  # Small penalty for each step
            done = False

        return self.agent_pos, reward, done, {}

    def render(self, mode='human'):
        # Simple rendering: print agent's position in the 3D grid
        grid = np.zeros(self.grid_size)
        grid[tuple(self.agent_pos)] = 1  # Mark the agent's position
        print("Agent's position:\n", self.agent_pos)

    def close(self):
        pass

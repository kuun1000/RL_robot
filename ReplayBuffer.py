import numpy as np
import random
from collections import deque

# Replay Buffer 정의
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state_rgbd, state_motor, action, reward, next_state_rgbd, next_state_motor, done):
        self.buffer.append((state_rgbd, state_motor, action, reward, next_state_rgbd, next_state_motor, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state_rgbd, state_motor, action, reward, next_state_rgbd, next_state_motor, done = map(np.stack, zip(*batch))
        return state_rgbd, state_motor, action, reward, next_state_rgbd, next_state_motor, done

    def __len__(self):
        return len(self.buffer)
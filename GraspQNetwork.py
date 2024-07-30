import torch
import torch.nn as nn
import numpy as np

# Grasp-Q-Network 정의
class GraspQNetwork(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(GraspQNetwork, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(self._get_conv_output(input_shape), 512)
        self.fc2 = nn.Linear(512, num_actions)

    def _get_conv_output(self, shape):
        o = torch.zeros(1, *shape)
        o = self.conv1(o)
        o = self.conv2(o)
        o = self.conv3(o)
        return int(np.prod(o.size()))

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)
    


# Q-Network 학습 관련 함수
def update_target(q_network, target_q_network):
    target_q_network.load_state_dict(q_network.state_dict())

def compute_td_loss(replay_buffer, batch_size, q_network, target_q_network, gamma, optimizer):
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)
    state = torch.tensor(state).float().cuda()
    next_state = torch.tensor(next_state).float().cuda()
    action = torch.tensor(action).long().cuda()
    reward = torch.tensor(reward).float().cuda()
    done = torch.tensor(done).float().cuda()

    q_values = q_network(state)
    next_q_values = target_q_network(next_state)

    q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    next_q_value = next_q_values.max(1)[0]
    expected_q_value = reward + gamma * next_q_value * (1 - done)

    loss = (q_value - expected_q_value.detach()).pow(2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss
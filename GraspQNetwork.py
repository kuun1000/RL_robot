import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Grasp-Q-Network 정의
class GraspQNetwork(nn.Module):
    def __init__(self, input_shape_rgbd, motor_input_dim, num_actions):
        super(GraspQNetwork, self).__init__()
        
        # RGB-D 이미지 처리
        self.conv1_rgbd = nn.Conv2d(input_shape_rgbd[0], 32, kernel_size=7, stride=2)
        self.conv2_rgbd = nn.Conv2d(32, 64, kernel_size=5, stride=2)
        self.conv3_rgbd = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        self.bn1_rgbd = nn.BatchNorm2d(32)
        self.bn2_rgbd = nn.BatchNorm2d(64)
        self.bn3_rgbd = nn.BatchNorm2d(64)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        # Conv 출력 크기 계산
        conv_out_size_rgbd = self._get_conv_output(input_shape_rgbd)
        
        # Fully connected layers for RGB-D
        self.fc1_rgbd = nn.Linear(conv_out_size_rgbd, 512)
        
        # Fully connected layers for motor input
        self.fc1_motor = nn.Linear(motor_input_dim, 128)
        self.fc2_motor = nn.Linear(128, 64)
        
        # Merge and final layers
        self.fc_merge = nn.Linear(512 + 64, 512)
        self.fc_final = nn.Linear(512, num_actions)

    def _get_conv_output(self, shape):
        o = torch.zeros(1, *shape)
        o = self.pool(F.relu(self.bn1_rgbd(self.conv1_rgbd(o))))
        o = self.pool(F.relu(self.bn2_rgbd(self.conv2_rgbd(o))))
        o = F.relu(self.bn3_rgbd(self.conv3_rgbd(o)))
        return int(np.prod(o.size()))

    def forward(self, x_rgbd, x_motor):
        # RGB-D 이미지 처리
        x_rgbd = self.pool(F.relu(self.bn1_rgbd(self.conv1_rgbd(x_rgbd))))
        x_rgbd = self.pool(F.relu(self.bn2_rgbd(self.conv2_rgbd(x_rgbd))))
        x_rgbd = F.relu(self.bn3_rgbd(self.conv3_rgbd(x_rgbd)))
        x_rgbd = x_rgbd.reshape(x_rgbd.size(0), -1)
        x_rgbd = F.relu(self.fc1_rgbd(x_rgbd))
        
        # 모터 입력 처리
        x_motor = F.relu(self.fc1_motor(x_motor))
        x_motor = F.relu(self.fc2_motor(x_motor))
        
        # 결합
        x = torch.cat((x_rgbd, x_motor), dim=1)
        x = F.relu(self.fc_merge(x))

        return self.fc_final(x)
    


# Q-Network 학습 관련 함수
def update_target(q_network, target_q_network):
    target_q_network.load_state_dict(q_network.state_dict())

def compute_td_loss(replay_buffer, batch_size, q_network, target_q_network, gamma, optimizer):
    state_rgbd, state_motor, action, reward, next_state_rgbd, next_state_motor, done = replay_buffer.sample(batch_size)
    state_rgbd = torch.tensor(state_rgbd).float().cuda()
    state_motor = torch.tensor(state_motor).float().cuda()
    next_state_rgbd = torch.tensor(next_state_rgbd).float().cuda()
    next_state_motor = torch.tensor(next_state_motor).float().cuda()
    action = torch.tensor(action).long().cuda()
    reward = torch.tensor(reward).float().cuda()
    done = torch.tensor(done).float().cuda()

    state_rgbd = state_rgbd.permute(0, 3, 1, 2)  # (batch_size, channels, height, width)
    next_state_rgbd = next_state_rgbd.permute(0, 3, 1, 2)  # (batch_size, channels, height, width)

    q_values = q_network(state_rgbd, state_motor)   # (batch_size, num_actions)
    # print(f"shape of q_values:{q_values.shape}")
    # print(f"number of action: {action.shape}")
    next_q_values = target_q_network(next_state_rgbd, next_state_motor)

    # Convert action to appropriate shape
    q_value = torch.sum(q_values * action, dim=1)
    # action = action.reshape(-1, 1)
    # print(f"action shape: {action.shape}")

    # q_value = q_values.gather(1, action).squeeze(1)
    next_q_value = next_q_values.max(1)[0]
    expected_q_value = reward + gamma * next_q_value * (1 - done)

    loss = (q_value - expected_q_value.detach()).pow(2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss
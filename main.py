import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from ReplayBuffer import ReplayBuffer
from xArm_Env import xArmEnv

# 환경 초기화 및 학습 설정
env = xArmEnv()
state_dim = env.observation_space.spaces['joint_angles'].shape[0]  # joint_angles의 크기
num_actions = env.action_space.spaces['end_effector_position'].shape[0] + \
              env.action_space.spaces['end_effector_rotation'].shape[0] + \
              env.action_space.spaces['gripper_action'].shape[0]

# A2C 네트워크 정의
class A2C(nn.Module):
    def __init__(self, state_dim, num_actions, hidden_size=128):
        super(A2C, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_actions),
            nn.Softmax(dim=-1)  # 행동 확률 분포를 출력
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)  # 상태 가치값 출력
        )

    def forward(self, state):
        action_prob = self.actor(state)
        value = self.critic(state)
        return action_prob, value

# 네트워크 및 최적화 설정
a2c = A2C(state_dim, num_actions).cuda()
optimizer = optim.Adam(a2c.parameters(), lr=1e-4)
gamma = 0.99

# 학습 함수 정의
def compute_loss(rewards, log_probs, values):
    # 각 타임 스텝에서의 Advantage 계산
    Q_vals = []
    Q_val = 0
    for reward in reversed(rewards):
        Q_val = reward + gamma * Q_val
        Q_vals.insert(0, Q_val)

    Q_vals = torch.tensor(Q_vals).float().cuda()
    values = torch.stack(values).squeeze()
    log_probs = torch.stack(log_probs)

    advantage = Q_vals - values

    actor_loss = -(log_probs * advantage.detach()).mean()
    critic_loss = advantage.pow(2).mean()

    return actor_loss + critic_loss

# 학습 루프
num_episodes = 1000
max_steps_per_episode = 200
for episode in range(num_episodes):
    state = torch.tensor(env.reset()['joint_angles']).float().cuda().unsqueeze(0)
    log_probs = []
    values = []
    rewards = []
    episode_reward = 0

    for step in range(max_steps_per_episode):
        action_probs, value = a2c(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        # action을 환경에 전달하여 다음 상태, 보상, 종료 여부를 얻음
        next_state, reward, done = env.step(action.cpu().numpy())
        next_state = torch.tensor(next_state['joint_angles']).float().cuda().unsqueeze(0)

        log_probs.append(log_prob)
        values.append(value)
        rewards.append(reward)
        state = next_state
        episode_reward += reward

        if done:
            break

    # 손실 계산 및 네트워크 업데이트
    loss = compute_loss(rewards, log_probs, values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Episode {episode}, Total Reward: {episode_reward}")

print("학습 완료!")

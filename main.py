import torch
import torch.optim as optim
from GraspQNetwork import GraspQNetwork, update_target, compute_td_loss
from ReplayBuffer import ReplayBuffer
from xArm_Env import xArm6GraspEnv

# 환경 초기화 및 학습 설정
env = xArm6GraspEnv()
input_shape_rgbd = (4, 480, 640)  # RGB-D 이미지의 입력 형태
motor_input_dim = env.observation_space.spaces['joint_angles'].shape[0]
num_actions = env.action_space.spaces['end_effector_position'].shape[0] + \
              env.action_space.spaces['end_effector_rotation'].shape[0] + \
              env.action_space.spaces['gripper_action'].shape[0]  # 행동 공간의 크기



# 네트워크 및 최적화 설정
q_network = GraspQNetwork(input_shape_rgbd, motor_input_dim, num_actions).cuda()
target_q_network = GraspQNetwork(input_shape_rgbd, motor_input_dim, num_actions).cuda()
optimizer = optim.Adam(q_network.parameters())
replay_buffer = ReplayBuffer(10000)
batch_size = 32
gamma = 0.99



def convert_action(action_vector):
    end_effector_position = action_vector[:3]
    end_effector_rotation = action_vector[3:4]
    gripper_action = action_vector[4:5]
    return {
        'end_effector_position': end_effector_position,
        'end_effector_rotation': end_effector_rotation,
        'gripper_action': gripper_action
    }



# 학습 루프
num_episodes = 1000
max_steps_per_episode = 200  # 최대 스텝 수
update_target_steps = 100
for episode in range(num_episodes):
    state = env.reset()
    episode_reward = 0
    done = False
    step_count = 0

    while not done and step_count < max_steps_per_episode:
        step_count += 1
        rgbd_tensor = torch.tensor(state['rgbd']).float().cuda()
        rgbd_tensor = rgbd_tensor.permute(2, 0, 1).unsqueeze(0)  # (batch_size, channels, height, width)
        motor_tensor = torch.tensor(state['joint_angles']).float().cuda().unsqueeze(0)

        action_vector = q_network(rgbd_tensor, motor_tensor).detach().cpu().numpy().squeeze()
        action = convert_action(action_vector)
        next_state, reward, done = env.step(action)
        replay_buffer.push(state['rgbd'], state['joint_angles'], action_vector, reward, next_state['rgbd'], next_state['joint_angles'], done)
        state = next_state
        episode_reward += reward

        print(f"Episode: {episode}, Step: {step_count}, Reward: {reward}, Done: {done}, Episode Reward: {episode_reward}")

        if len(replay_buffer) > batch_size:
            loss = compute_td_loss(replay_buffer, batch_size, q_network, target_q_network, gamma, optimizer)

        if episode % update_target_steps == 0:
            update_target(q_network, target_q_network)

    print(f"Episode: {episode}, Total Steps: {step_count}, Reward: {episode_reward}")

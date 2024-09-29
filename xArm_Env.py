# xArm_Env.py 파일 수정
import gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
from makeCustomEnv import Grid3DEnv  # makeCustomEnv 파일에서 Grid3DEnv 클래스를 가져옴


class xArmEnv(gym.Env):
    def __init__(self, grid_size=(10, 10, 10)):
        super(xArmEnv, self).__init__()

        # Grid3DEnv 환경을 생성하여 목표 위치를 관리
        self.grid_env = Grid3DEnv(grid_size=grid_size, random_goal=True)

        # 3D Grid 환경 정의
        self.grid_size = grid_size  # Grid3DEnv와 호환되는 크기
        self.state = np.zeros(grid_size)
        self.agent_pos = np.array([0, 0, 0])
        self.goal_pos = self.grid_env.goal_pos  # Grid3DEnv에서 설정된 목표 위치로 설정

        # 관절 수와 행동 공간 정의
        self.num_joints = 7
        self.observation_space = spaces.Dict({
            'joint_angles': spaces.Box(low=-np.pi, high=np.pi, shape=(self.num_joints,), dtype=np.float32),
            'end_effector_position': spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),  # x, y, z 좌표
            'goal_position': spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),  # 목표 위치
        })

        # 행동 공간 정의
        self.action_space = spaces.Dict({
            'end_effector_position': spaces.Box(low=np.array([-1.0, -1.0, -1.0]), high=np.array([1.0, 1.0, 1.0]), dtype=np.float32),
            'end_effector_rotation': spaces.Box(low=np.array([-np.pi]), high=np.array([np.pi]), dtype=np.float32),
            'gripper_action': spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
        })

        # PyBullet 환경 설정
        self.client = None
        self.reset()

    def reset(self):
        if self.client is None:
            self.client = p.connect(p.GUI)
            p.setAdditionalSearchPath(pybullet_data.getDataPath())

        p.resetSimulation(self.client)
        p.setGravity(0, 0, -9.8, self.client)

        # 테이블 및 로봇 로드
        self.table_id = p.loadURDF("table/table.urdf",
                                   basePosition=[0, 0, 0],
                                   baseOrientation=p.getQuaternionFromEuler([0, 0, np.pi / 2]),
                                   physicsClientId=self.client)
        self.robot_id = p.loadURDF("lite_6_robotarm.urdf",
                                   basePosition=[-0.45, -0.05, 0.60],
                                   useFixedBase=True)

        self.ee = 6  # End-effector link index
        self.agent_pos = np.array([0, 0, 0])  # 초기 에이전트 위치

        # Grid3DEnv를 초기화하고 목표 위치 동기화
        self.grid_env.reset()
        self.goal_pos = self.grid_env.goal_pos

        return self._get_observation()

    def step(self, action):
        # End-effector의 위치를 액션으로 적용
        end_effector_pos_delta = action['end_effector_position']
        end_effector_pos = p.getLinkState(self.robot_id, self.ee)[0]
        new_pos = np.array(end_effector_pos) + np.array(end_effector_pos_delta)

        # Inverse Kinematics를 사용하여 로봇 관절 제어
        jointPoses = p.calculateInverseKinematics(self.robot_id, self.ee, new_pos)
        for i in range(self.num_joints):
            p.setJointMotorControl2(bodyIndex=self.robot_id,
                                    jointIndex=i,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=jointPoses[i])

        # Step 시뮬레이션 진행
        p.stepSimulation(self.client)

        # 현재 end-effector 위치 확인
        cur_end_effector_pos = p.getLinkState(self.robot_id, self.ee)[0]

        # Grid3DEnv에서 에이전트 위치를 업데이트하고 목표 도달 여부 확인
        self.grid_env.agent_pos = np.array(cur_end_effector_pos)
        _, grid_reward, grid_done, _, _ = self.grid_env.step(self._convert_action_to_grid(action))

        # 목표에 도달했는지 확인 (Grid3DEnv와 일치)
        done = grid_done
        reward = 1.0 if done else -0.1  # 목표 도달 시 보상, 아니면 패널티

        # 상태 관측 값 반환
        obs = self._get_observation()
        return obs, reward, done, False, {}

    def _get_observation(self):
        """현재 관절 각도와 end-effector 위치를 반환."""
        joint_states = p.getJointStates(self.robot_id, range(self.num_joints))
        joint_positions = np.array([state[0] for state in joint_states])

        end_effector_pos = p.getLinkState(self.robot_id, self.ee)[0]

        observation = {
            'joint_angles': joint_positions,
            'end_effector_position': np.array(end_effector_pos),
            'goal_position': self.goal_pos
        }

        return observation

    def _convert_action_to_grid(self, action):
        """xArm의 행동을 Grid3DEnv의 행동으로 변환."""
        end_effector_pos_delta = action['end_effector_position']
        # x, y, z 축 중 이동이 큰 방향으로 action을 결정 (0-5 사이)
        move_index = np.argmax(np.abs(end_effector_pos_delta))
        direction = np.sign(end_effector_pos_delta[move_index])
        if move_index == 0:  # x 축 이동
            return 0 if direction > 0 else 1
        elif move_index == 1:  # y 축 이동
            return 2 if direction > 0 else 3
        elif move_index == 2:  # z 축 이동
            return 4 if direction > 0 else 5

    def render(self, mode='human'):
        self.grid_env.render()

    def close(self):
        if self.client is not None:
            p.disconnect(self.client)
            self.client = None
        self.grid_env.close()


# 환경 테스트
if __name__ == "__main__":
    # xArm 환경을 생성하고 초기화
    env = xArmEnv(grid_size=(10, 10, 10))
    obs, _ = env.reset()
    print("Initial Observation:", obs)

    done = False
    while not done:
        action = {'end_effector_position': np.random.uniform(-0.1, 0.1, size=(3,)),
                  'end_effector_rotation': np.array([0.0]),
                  'gripper_action': np.array([0.0])}
        obs, reward, done, _, _ = env.step(action)
        print(f"Action: {action}, Observation: {obs}, Reward: {reward}, Done: {done}")

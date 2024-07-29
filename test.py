import pybullet as p
import pybullet_data
import os

# PyBullet 시뮬레이션 초기화
p.connect(p.GUI)

# xArm 6 URDF 파일 로드
robot_id = p.loadURDF("lite_6_robotarm.urdf", useFixedBase=True)
# robot_id = p.loadURDF(os.path.join(pybullet_data.getDataPath(), "franka_panda/panda.urdf"), useFixedBase=True)

# 조인트 정보 출력
num_joints = p.getNumJoints(robot_id)
for joint_index in range(num_joints):
    joint_info = p.getJointInfo(robot_id, joint_index)
    print(f"Joint {joint_index}: {joint_info[1]}")
    

# PyBullet 종료
p.disconnect()
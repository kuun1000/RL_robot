import pybullet as p
import pybullet_data
import os
import math
import random
import numpy as np
import cv2

p.connect(p.GUI)
p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=0, cameraPitch=-20, cameraTargetPosition=[0.55,-0.35,0.1])
p.setGravity(0,0,-9.8)
pi = math.pi

# 로봇 팔, 테이블, 트레이박스 로드
arm = p.loadURDF("lite_6_robotarm.urdf", [0,0,0], p.getQuaternionFromEuler([0,0,-pi/2]), flags=p.URDF_USE_INERTIA_FROM_FILE, useFixedBase=True)
table = p.loadURDF(os.path.join(pybullet_data.getDataPath(),"table/table.urdf"), basePosition=[0.5,0,-0.67])
traybox = p.loadURDF(os.path.join(pybullet_data.getDataPath(),"tray/traybox.urdf"), basePosition=[0.65,0,0])

# 트레이박스 크기 및 중심 위치 가져오기
tray_aabb = p.getAABB(traybox)
tray_min = tray_aabb[0]
tray_max = tray_aabb[1]
tray_length = tray_max[0] - tray_min[0]
tray_width = tray_max[1] - tray_min[1]
tray_center = [(tray_min[0] + tray_max[0]) / 2, (tray_min[1] + tray_max[1]) / 2, (tray_min[2] + tray_max[2]) / 2]
margin = 0.05

# 카메라 설정
fov, aspect, near, far = 60, 1.0, 0.01, 100
projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)

# 실시간 시뮬레이션 설정
useRealTimeSimulation = 1
p.setRealTimeSimulation(useRealTimeSimulation)

# IK를 사용하여 로봇 팔의 그리퍼를 트레이 중앙으로 이동
gripper_target_pos = [tray_center[0], tray_center[1], tray_center[2] + 0.1]

# 그리퍼가 바닥을 향하도록 하는 방향 (Z축을 따라 아래로)
gripper_target_orientation = p.getQuaternionFromEuler([0, 0, 0])

# 그리퍼 목표 위치로의 역기구학 계산
joint_positions = p.calculateInverseKinematics(arm, 8, gripper_target_pos, gripper_target_orientation)

# 큐브 생성 함수
def create_cubes(num_cubes=5):
    cube_ids = []
    for _ in range(num_cubes):
        random_x = tray_center[0] + (random.uniform(-tray_length / 2 + margin, tray_length / 2 - margin))
        random_y = tray_center[1] + (random.uniform(-tray_width / 2 + margin, tray_width / 2 - margin))
        cube_start_pos = [random_x, random_y, tray_center[2] + 0.1]
        cube_start_orientation = p.getQuaternionFromEuler([0, 0, 0])
        cube_id = p.loadURDF(os.path.join(pybullet_data.getDataPath(), "cube_small.urdf"), cube_start_pos, cube_start_orientation)
        cube_ids.append(cube_id)
    return cube_ids

# 로봇 팔 카메라 함수
def arm_camera():
    # Center of mass position and orientation(of link-9)
    com_p, com_o, _, _, _, _ = p.getLinkState(arm, 9)

    # Camera setting(fov: 시야각, aspect: 종횡비)
    height, width = 480, 640
    fov, aspect, near, far = 60, width/height, 0.01, 15
    projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)    # 카메라 시점, 원근감 표현

    rot_matrix = p.getMatrixFromQuaternion(com_o)
    rot_matrix = np.array(rot_matrix).reshape(3,3)

    # Initial vectors
    init_camera_vector = (0, 0, -1) # z-axis
    init_up_vector = (0, 1, 0)  # y-axis

    # Rotated vectors
    camera_vector = rot_matrix.dot(init_camera_vector)
    up_vector = rot_matrix.dot(init_up_vector)
    view_matrix = p.computeViewMatrix(com_p, com_p + 1.0 * camera_vector, up_vector)    # 카메라 위치, 방향 이동

    img = p.getCameraImage(width, height, view_matrix, projection_matrix,
                           shadow=True,
                           renderer=p.ER_BULLET_HARDWARE_OPENGL)
    
    rgb_opengl = np.reshape(img[2], (height, width, 4))*(1./255.)
    rgb_opengl_uint8 = np.array(rgb_opengl * 255, dtype=np.uint8)
    rgb_img = cv2.cvtColor(rgb_opengl_uint8, cv2.COLOR_RGB2BGR)

    depth_opengl = np.reshape(img[3],(height, width))
    depth_img_normalized = cv2.normalize(depth_opengl, None, 0, 255, cv2.NORM_MINMAX)
    depth_img = np.uint8(depth_img_normalized)

    segmentation_image = np.array(img[4]).reshape((height, width))
    seg_img = cv2.applyColorMap(np.uint8(segmentation_image * 255 / segmentation_image.max()), cv2.COLORMAP_JET)
    
    return rgb_img,depth_img,seg_img

create_cubes(5) # 트레이박스 내 랜덤 위치에 큐브 5개 생성

# 시뮬레이션 루프
while True:
    # p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)
    # p.setJointMotorControl2(arm, 0, p.POSITION_CONTROL, 0)
    # p.setJointMotorControl2(arm, 1, p.POSITION_CONTROL, -0.4)
    # p.setJointMotorControl2(arm, 2, p.POSITION_CONTROL, 0.5)
    
    # 
    p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)
    for joint_index in range(len(joint_positions)):
        p.setJointMotorControl2(arm, joint_index, p.POSITION_CONTROL, joint_positions[joint_index])

    p.stepSimulation()
    arm_camera()

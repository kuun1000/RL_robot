import time
import torch
import gymnasium as gym
from xArm_Env import xArmEnv
import cv2
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# YOLOv5 모델을 로드합니다. (custom training된 모델인 'best.pt' 파일 사용)
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')

# 환경을 초기화합니다.
env = xArmEnv()

# 환경을 리셋하여 초기 상태를 얻습니다.
observation = env.reset()

# 몇 번의 랜덤 스텝을 수행하며 환경을 렌더링합니다.
for _ in range(10000):
    # 0.1초 대기합니다.
    time.sleep(0.1)

    # 환경을 렌더링합니다.
    env.render()

    ##+++이미지 가져오기+++++
    rgb_img, depth_img = env.arm_camera()

    # YOLOv5를 사용하여 객체 감지
    rgb_img_resized = cv2.resize(rgb_img, (640, 640))
    results = model(rgb_img_resized)  # conf_thres는 직접 설정하지 않음
    # print(results)
    # print(model.names)

    # 결과에서 감지된 객체 정보 가져오기 (바운딩 박스와 레이블)
    # results.xyxy[0]: 좌표, confidence, class ID 등이 담긴 데이터
    labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]

    # 이미지에 바운딩 박스를 그리기
    for i in range(len(labels)):
        row = cord[i]
        if row[4] >= 0.1:
            x1, y1, x2, y2 = int(row[0] * rgb_img.shape[1]), int(row[1] * rgb_img.shape[0]), \
                             int(row[2] * rgb_img.shape[1]), int(row[3] * rgb_img.shape[0])
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            center = (cx, cy)
            # print(center)
            depth_value = depth_img[cy, cx]    # depth 추출
            
            label = model.names[int(labels[i])]

            cv2.rectangle(rgb_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # cv2.putText(rgb_img, f"{label} {row[4]:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.putText(rgb_img, f"{(cx, cy)}", (cx-40, cy-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
            cv2.putText(rgb_img, f"Depth: {depth_value:.2f}", (rgb_img.shape[1] - 130, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    # 감지된 객체들이 표시된 이미지를 출력
    cv2.imshow("Cube Detections", rgb_img)
    cv2.imshow("Depth Image", depth_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    ##++++++++

    # 0.1초 대기합니다.
    time.sleep(0.1)

# 환경을 종료합니다.
env.close()
cv2.destroyAllWindows()

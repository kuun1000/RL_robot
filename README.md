# 사용되는 파이썬 파일


* xArm6_PickandPlace_Env.py: xArm6 로봇을 가지고 물체를 집어올리는 환경
  * Rename해야 함(PickandPlace -> Pick)
  
* GraspQNetwork:  RGB-D 이미지와 motor 정보를 입력으로 받아 Q values를 출력하는 신경망

* ReplayBuffer: 리플레이 버퍼

* main: 학습
  * Bug: loss 계산하는 과정 문제 있음 -> 현재: 학습 이루어지지 않는 상태

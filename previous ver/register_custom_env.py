from gymnasium.envs.registration import register

register(
    id='CustomRobotPickAndPlace-v5',
    entry_point='pick_and_place:PickAndPlace',  # 파일 경로와 클래스 이름을 적절히 수정
)
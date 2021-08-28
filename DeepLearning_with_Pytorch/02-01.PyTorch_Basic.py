# 02
# 01 파이토치 패키지의 기본 구성

'''

torch : 메인 네임 스페이스

torch.autograd : 자동 미분을 위한 함수들이 포함되어 있다.
    on / off 를 제어하는 콘텍스트 매니저 > enable_grad / no_grad
    미분 가능 함수를 정의할 떄 사용하는 기반 클래스 Funtion

torch.nn : 신경망을 구축하기 위한 다양한 데이터 구조나 레이어 등이 정의 되어져 있다.

torch.optim : 확률적 경사 하강법 SGD를 중심으로 최적화 알고리즘이 구현되어 있다.

torch.utils.data : 미니 배치용 유틸리티 함수가 포함되어 있다.

torch.onnx : ONNX(Open Neural Network Exchange), 모델을 export할 때 사용, 서로 다른 딥 런이 프레임워크 간에 모델을 공유할 때 사용

'''
# 1. 머신러닝 모델의 평가
"""
하이퍼파라미터 : 사용자가 직접 정해줄 수 있는 변수 (learning rate, # of nodes, dropout rate,,,)
매개변수 : 모델이 학습하는 과정에서 얻어지는 값, 기계가 훈련을 통해서 바꾸는 변수
"""

# 분류 / 회귀
"""
1) 이진 분류 문제
둘 중 하나의 답을 정하는 문제

2) 다중 클래스 분류
세 개 이상의 정해진 선택지 중에서 답을 정하는 문제

3) 회귀 문제
연속된 값을 결과로 가짐
"""

# 지도 학습 / 비지도 학습
"""
1) 지도 학습
레이블이라는 정답과 함께 학습하는 것

2) 비지도 학습
목적 데이터 (레이블) 없는 학습 방법
clustering, 차원 축소

3) 강화 학습
어떤 환경 내에서 정의된 에이전트가 현재의 상태를 인식하여, 선택 가능한 행동들 중 보상을 최대화하는 행동 혹은 행동 순서를 선택하는 방법
"""

# 샘플 / 특성
"""
하나의 데이터, 하나의 행 -> 샘플
종속 변수 y를 예측하기 위한 각각의 독립 변수 x -> 특성
"""

# 혼동 행렬 Confusiotn Matrix
"""
열 : 예측 값, 행 : 실제 값
    |   참   |   거짓
========================
참  |   TP   |   FN
거짓|   FP   |   TN

FP : 양성이라고 예측했는데, 실제로는 음성인 경우
FN : 음성이라고 예측했는데, 실제로는 양성인 경우

Precision : TP / TP + FP
Recall : TP / TP + FN
"""

# 과적합 / 과소 적합
"""
과적합 : 훈련 데이터를 과하게 학습한 경우 -> 테스트 데이터나 실제 서비스에서의 데이터에 대해서는 정확도가 좋지 않은 현상이 발생함
과소적합 : 테스트 데이터의 성능이 올라갈 여지가 있음에도 훈련을 덜 한 상태  -> 훈련 데이터에 대한 정확도가 낮다.
"""
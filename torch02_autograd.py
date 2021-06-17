# 기본적인 딥러닝 모델
# Autograd 방식: back propagation을 이용해 파라미터를 업데이트 하는 방법을 구현해보자

import torch

if torch.cuda.is_available() :      # GPU를 이용해 계산할 수 있는지 파악하는 method
    DEVICE = torch.device('cuda')   # 참, cuda 이용
else :
    DEVICE = torch.device('cpu')    # 거짓, cpu 이용

print("DEVICE : ", DEVICE)  # DEVICE :  cuda

BATCH_SIZE = 64     # 파라미터를 업데이트할 때 계산되는 데이터의 개수
INPUT_SIZE = 1000   # 모델에서의 input 크기, 입력층의 노드 수
                    # BATCH_SIZE = 64 & INPUT_SIZE = 1000  --> 1000크기의 벡터 값을 64개 이용한다. (64, 1000)
HIDDEN_SIZE = 100   # 은닉층의 노드 수 (1000, 100)
OUTPUT_SIZE = 10    # 모델에서 최종으로 출력되는 값의 벡터 크기, 레이블의 크기와 동일하게 설정함

# randn : 평균 0, 표준편자 1인 정규분포에서 샘플링한 값, 데이터를 만든다는 것을 의미함
x = torch.randn(BATCH_SIZE,
                INPUT_SIZE,             # (64, 1000)
                device=DEVICE,  
                dtype=torch.float,      # 데이터 형태
                requires_grad=False)    # input으로 이용되기 때문에 gradient를 계산할 필요없다.

y = torch.randn(BATCH_SIZE,
                OUTPUT_SIZE,
                device=DEVICE,
                dtype=torch.float,
                requires_grad=False)

w1 = torch.randn(INPUT_SIZE,
                HIDDEN_SIZE,        # (1000, 100)
                device=DEVICE,
                dtype=torch.float,
                requires_grad=True) # gradient를 계산할 수 있도록 설정한다.

w2 = torch.randn(HIDDEN_SIZE,
                OUTPUT_SIZE,        # (100, 10)
                device=DEVICE,
                dtype=torch.float,
                requires_grad=True)

# 파라미터 업데이터 확인
learning_rate = 1e-6                        # 파라미터 업데이트 할 때 learning rate 만큼 곱한 값으로 gradient가 없데이트 된다.

for t in range(1, 501) :                    # 500번 반복
    y_pred = x.mm(w1).clamp(min=0).mm(w2)   # input x 와 w1의 행렬곱 -> clam() 비선형 함수, 렐루와 유사 -> w2와 행렬곱한 걸 y_pred에 저장

    loss = (y_pred - y).pow(2).sum()        # loss : 실제 값과 예측한 값의 차이를 제곱한 것의 합
    if t % 100 == 0 :
        print("Iteration : " , t, "\t", "Loss : ", loss.item())
    loss.backward()                         # 각 파라미터 값에 대해 Gradient를 계산하고 이를 통해 back propagation을 진행한다.

    with torch.no_grad() :                  # gradient 값을 고정한다.
        w1 -= learning_rate * w1.grad       # w1에 기존 w1의 gradient * lr 값을 뺀 거로 업데이트 한다. 
        w2 -= learning_rate * w2.grad

        w1.grad.zero_()                     # gradient 값을 0으로 설정한다. 다음 반복문에서 loss.backward()을 통해 gradient가 새로 계산된다.        
        w2.grad.zero_()

# Iteration :  100         Loss :  904.1318969726562
# Iteration :  200         Loss :  16.17711067199707
# Iteration :  300         Loss :  11.086380958557129
# Iteration :  400         Loss :  11.939188003540039
# Iteration :  500         Loss :  11.102775573730469


# nn.Module로 구현하는 선형회귀
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(2021)

# 데이터 선언
x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[2], [4], [6]])

# 모델 선언 및 초기화
# 단순 선형 모델이기 때문에, input_dim=1, output_dim=1

model = nn.Linear(1,1)

# 모델에는 가중치와 bias가 저장되어 있다. 
print(list(model.parameters()))
# 첫번째 값이 W, 두번째 값이 b -> 모두 랜덤 초기화 되어 있고 학습할 때 업데이트 되도록 requrires_grad=True로 설정되어 있다.
# [Parameter containing:
# tensor([[-0.7391]], requires_grad=True), Parameter containing:
# tensor([0.0268], requires_grad=True)]

# optimizer 설정
# model.parameters()에 저장되어 있는 W, b를 불러온다.
optimizer = optim.SGD(model.parameters(), lr=0.01)

nb_epochs = 2200
for epoch in range(nb_epochs+1) : 
    # hypothesis H(x) 정의 =====> 입력 x로부터 y를 얻는 것을 forward 연산이라고 한다.
    prediction = model(x_train)

    # cost 계산
    # F.mse_loss : 평균 제곱 오차
    cost = F.mse_loss(prediction, y_train)

    optimizer.zero_grad()
    cost.backward() # ======> cost 함수를 미분하여 기울기를 구하는 것을 backward 연산이라고 한다.
    optimizer.step()

    if epoch % 100 == 0 : 
        print("Epoch {:4d}/{} | Cost {:.6f}".format(
            epoch, nb_epochs, cost.item()
        ))

# Epoch    0/2200 | Cost 34.720383
# Epoch  100/2200 | Cost 0.094413
# Epoch  200/2200 | Cost 0.058342
# Epoch  300/2200 | Cost 0.036052
# Epoch  400/2200 | Cost 0.022278
# Epoch  500/2200 | Cost 0.013766
# Epoch  600/2200 | Cost 0.008507
# Epoch  700/2200 | Cost 0.005257
# Epoch  800/2200 | Cost 0.003248
# Epoch  900/2200 | Cost 0.002007
# Epoch 1000/2200 | Cost 0.001240
# Epoch 1100/2200 | Cost 0.000766
# Epoch 1200/2200 | Cost 0.000474
# Epoch 1300/2200 | Cost 0.000293
# Epoch 1400/2200 | Cost 0.000181
# Epoch 1500/2200 | Cost 0.000112
# Epoch 1600/2200 | Cost 0.000069
# Epoch 1700/2200 | Cost 0.000043
# Epoch 1800/2200 | Cost 0.000026
# Epoch 1900/2200 | Cost 0.000016
# Epoch 2000/2200 | Cost 0.000010
# Epoch 2100/2200 | Cost 0.000006
# Epoch 2200/2200 | Cost 0.000004

# Predict
new_var = torch.FloatTensor([[4.0]])
pred_y = model(new_var)
print("훈련 후, 입력이 4일 때 예측값 : ", pred_y)
# 훈련 후, 입력이 4일 때 예측값 :  tensor([[7.9961]], grad_fn=<AddmmBackward>)

# 학습후, W와 b를 출력해보자
print(list(model.parameters()))
# [Parameter containing:
# tensor([[1.9977]], requires_grad=True), Parameter containing:
# tensor([0.0052], requires_grad=True)
# W는 거의 2에 가깝고 b는 거의 0에 가깝다.



# nn.Module로 구현하는 선형회귀
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(2021)

# 데이터 선언
x_train = torch.FloatTensor([[73, 80, 75],
                             [93, 88, 93],
                             [89, 91, 80],
                             [96, 98, 100],
                             [73, 66, 70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])

# 모델 선언 및 초기화
# 다중 선형 모델이기 때문에, input_dim=3, output_dim=1
model = nn.Linear(3,1)

# 모델에는 가중치와 bias가 저장되어 있다. 
print(list(model.parameters()))
# 첫번째 값이 W, 두번째 값이 b -> 모두 랜덤 초기화 되어 있고 학습할 때 업데이트 되도록 requrires_grad=True로 설정되어 있다.
# 입력값이 3개이므로 w가 3개 저장되어 있다. 
# [Parameter containing:
# tensor([[-0.4267,  0.0155,  0.2801]], requires_grad=True), Parameter containing:
# tensor([0.2493], requires_grad=True)]

# optimizer 설정
# model.parameters()에 저장되어 있는 W, b를 불러온다.
optimizer = optim.SGD(model.parameters(), lr=1e-5)

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

# Epoch    0/2200 | Cost 33670.605469
# Epoch  100/2200 | Cost 15.317639
# Epoch  200/2200 | Cost 14.687892
# Epoch  300/2200 | Cost 14.086802
# Epoch  400/2200 | Cost 13.512731
# Epoch  500/2200 | Cost 12.964491
# Epoch  600/2200 | Cost 12.440659
# Epoch  700/2200 | Cost 11.940115
# Epoch  800/2200 | Cost 11.461645
# Epoch  900/2200 | Cost 11.004146
# Epoch 1000/2200 | Cost 10.566730
# Epoch 1100/2200 | Cost 10.148247
# Epoch 1200/2200 | Cost 9.747847
# Epoch 1300/2200 | Cost 9.364656
# Epoch 1400/2200 | Cost 8.997896
# Epoch 1500/2200 | Cost 8.646748
# Epoch 1600/2200 | Cost 8.310492
# Epoch 1700/2200 | Cost 7.988463
# Epoch 1800/2200 | Cost 7.679895
# Epoch 1900/2200 | Cost 7.384245
# Epoch 2000/2200 | Cost 7.100915
# Epoch 2100/2200 | Cost 6.829389
# Epoch 2200/2200 | Cost 6.568987

# Predict
new_var = torch.FloatTensor([[73, 80, 75]])
pred_y = model(new_var)
print("훈련 후, 입력이 73, 80, 75일 때 예측값 : ", pred_y)
# 훈련 후, 입력이 73, 80, 75일 때 예측값 :  tensor([[154.7190]], grad_fn=<AddmmBackward>)

# 학습후, W와 b를 출력해보자
print(list(model.parameters()))
# [Parameter containing:
# tensor([[0.5947, 0.7704, 0.6588]], requires_grad=True), Parameter containing:
# tensor([0.2611], requires_grad=True)]



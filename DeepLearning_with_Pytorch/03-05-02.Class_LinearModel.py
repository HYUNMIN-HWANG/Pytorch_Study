# 클래스로 모델 구현하기
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(2021)

x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[2], [4], [6]])

# 단순 선형 회귀 모델
class LinearRegressionModel(nn.Module): # nn.Module을 상속받는 파이썬 클래스
    def __init__(self):
        super().__init__()              # super() : nn.Module 클래스의 속성들을 가지고 초기화함
        self.linear = nn.Linear(1,1)    # input dim=1, output dim=1

    def forward(self, x):               # forward 연산 : model이란 객체를 생성한 후,model에 입력 데이터를 넣으면 객체를 호출하면서 해당 함수를 실행시킨다.
        return self.linear(x)

model = LinearRegressionModel()

optimizer = optim.SGD(model.parameters(), lr=0.01)

nb_epochs = 2000
for epoch in range(nb_epochs+1):
    # hypothesis
    prediction = model(x_train)

    cost = F.mse_loss(prediction, y_train)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print("Epoch {:4d} / {} | Cost : {:.6f}".format(
            epoch, nb_epochs, cost.item()
        ))

# Epoch    0 / 2000 | Cost : 34.720383
# Epoch  100 / 2000 | Cost : 0.094413
# Epoch  200 / 2000 | Cost : 0.058342
# Epoch  300 / 2000 | Cost : 0.036052
# Epoch  400 / 2000 | Cost : 0.022278
# Epoch  500 / 2000 | Cost : 0.013766
# Epoch  600 / 2000 | Cost : 0.008507
# Epoch  700 / 2000 | Cost : 0.005257
# Epoch  800 / 2000 | Cost : 0.003248
# Epoch  900 / 2000 | Cost : 0.002007
# Epoch 1000 / 2000 | Cost : 0.001240
# Epoch 1100 / 2000 | Cost : 0.000766
# Epoch 1200 / 2000 | Cost : 0.000474
# Epoch 1300 / 2000 | Cost : 0.000293
# Epoch 1400 / 2000 | Cost : 0.000181
# Epoch 1500 / 2000 | Cost : 0.000112
# Epoch 1600 / 2000 | Cost : 0.000069
# Epoch 1700 / 2000 | Cost : 0.000043
# Epoch 1800 / 2000 | Cost : 0.000026
# Epoch 1900 / 2000 | Cost : 0.000016
# Epoch 2000 / 2000 | Cost : 0.000010

# 클래스로 모델 구현하기
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(2021)

x_train = torch.FloatTensor([[73, 80, 75],
                             [93, 88, 93],
                             [89, 91, 80],
                             [96, 98, 100],
                             [73, 66, 70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])


# 단순 선형 회귀 모델
class MultivariateLinearRegression(nn.Module): # nn.Module을 상속받는 파이썬 클래스
    def __init__(self):
        super().__init__()              # super() : nn.Module 클래스의 속성들을 가지고 초기화함
        self.linear = nn.Linear(3,1)    # input dim=1, output dim=1

    def forward(self, x):               # forward 연산 : model이란 객체를 생성한 후,model에 입력 데이터를 넣으면 객체를 호출하면서 해당 함수를 실행시킨다.
        return self.linear(x)

model = MultivariateLinearRegression()

optimizer = optim.SGD(model.parameters(), lr=1e-5)

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

# Epoch    0 / 2000 | Cost : 33670.605469
# Epoch  100 / 2000 | Cost : 15.317639
# Epoch  200 / 2000 | Cost : 14.687892
# Epoch  300 / 2000 | Cost : 14.086802
# Epoch  400 / 2000 | Cost : 13.512731
# Epoch  500 / 2000 | Cost : 12.964491
# Epoch  600 / 2000 | Cost : 12.440659
# Epoch  700 / 2000 | Cost : 11.940115
# Epoch  800 / 2000 | Cost : 11.461645
# Epoch  900 / 2000 | Cost : 11.004146
# Epoch 1000 / 2000 | Cost : 10.566730
# Epoch 1100 / 2000 | Cost : 10.148247
# Epoch 1200 / 2000 | Cost : 9.747847
# Epoch 1300 / 2000 | Cost : 9.364656
# Epoch 1400 / 2000 | Cost : 8.997896
# Epoch 1500 / 2000 | Cost : 8.646748
# Epoch 1600 / 2000 | Cost : 8.310492
# Epoch 1700 / 2000 | Cost : 7.988463
# Epoch 1800 / 2000 | Cost : 7.679895
# Epoch 1900 / 2000 | Cost : 7.384245
# Epoch 2000 / 2000 | Cost : 7.100915


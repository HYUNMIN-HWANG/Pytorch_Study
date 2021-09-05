# 클래스로 모델 구현하기
import torch
from torch._C import LiteScriptModule
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(2021)

# 1) 단순 선형 회귀 모델
class LinearRegressionModel(nn.Module): # nn.Module을 상속받는 파이썬 클래스
    def __init__(self):
        super().__init__()              # super() : nn.Module 클래스의 속성들을 가지고 초기화함
        self.linear = nn.Linear(1,1)    # input dim=1, output dim=1

    def forward(self, x):               # forward 연산 : model이란 객체를 생성한 후,model에 입력 데이터를 넣으면 객체를 호출하면서 해당 함수를 실행시킨다.
        return self.linear(x)

model = LinearRegressionModel()

# 2) 다중 선형 회귀 모델
class MultivariateLinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Lienar(3,1)

    def forward(self, x):
        return self.linear(x)

model = MultivariateLinearRegression()

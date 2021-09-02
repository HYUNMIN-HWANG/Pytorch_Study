import torch
from torch.autograd import grad_mode
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

# 다음에 다시 실행해도 같은 결과가 나오도록 시드를 고정시킨다.
torch.manual_seed(21)

'''1. DATA'''
x_train = torch.FloatTensor([[1],[2],[3]])  # 입력
y_train = torch.FloatTensor([[2],[4],[6]])  # 출력

print(x_train.shape)
print(y_train.shape)
# torch.Size([3, 1])
# torch.Size([3, 1])

'''2. hypothesis'''
# hypothesos : y = w * x + b ( w: 가중치, b: 편향)
# 가중치 w 초기화, 학습을 통해 값이 변경되는 변수임을 명시함
# requires_grad=True : 자동 미분 기능
W = torch.zeros(1, requires_grad=True)
print(W)
# tensor([0.], requires_grad=True)

# 편향 b 초기화
b = torch.zeros(1, requires_grad=True)
print(b)
# tensor([0.], requires_grad=True)

hypothesis = x_train * W + b
print(hypothesis)
# tensor([[0.],
#         [0.],
#         [0.]], grad_fn=<AddBackward0>)

'''3. Cost Function'''
# MSE가 가장 작은 직선을 찾는다.
cost = torch.mean((hypothesis - y_train) ** 2)
print(cost)
# tensor(18.6667, grad_fn=<MeanBackward0>)

'''4. Optimizer'''
# Optimizer 경사 하강법 : cost가 최소가 되는 지점은 W와 cost의 관계를 나타내는 그래프에서 가장 아래 쪽에 위치함 -> 접선의 기울기가 낮은 방향(0)으로 W의 값을 변경하는 작업을 반복한다.
# learning rate : W의 값을 변경할 때 얼마나 크게 변경할지를 결정

optimizer = optim.SGD([W, b], lr = 0.01)
print(optimizer)
# SGD (
# Parameter Group 0
#     dampening: 0
#     lr: 0.01
#     momentum: 0
#     nesterov: False
#     weight_decay: 0
# )

# 기울기를 0으로 초기화함  (why? 새로운 가중치 편향에 새로운 기울기를 구할 수 있다.) 
optimizer.zero_grad()
print(optimizer)

# 비용 함수를 미분하여 gradient(기울기) 계산
cost.backward()

# W와 b 업데이트
optimizer.step()







# 다중 선형 회귀 : 다수의 x로부터 y를 예측

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(2021)

'''Data'''
# 데이터 선언
x1_train = torch.FloatTensor([[73], [93], [89], [96], [73]])
x2_train = torch.FloatTensor([[80], [88], [91], [98], [66]])
x3_train = torch.FloatTensor([[75], [93], [90], [100], [70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])

# W, b 초기화
w1 = torch.zeros(1, requires_grad=True)
w2 = torch.zeros(1, requires_grad=True)
w3 = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# optimizer
optimizer = optim.SGD([w1, w2, w3, b], lr=1e-5)

np_epochs = 2999
for epoch in range(np_epochs + 1) :
    # 데이터 개수만큼 W를 곱해준다.
    # 단점 : 데이터의 개수가 많아지면 그만큼의 x와 w값을 정의해 줘야 한다. -> 비효울적임
    hypothesis = x1_train * w1 + x2_train * w2 + x3_train * w3 + b

    cost = torch.mean((hypothesis-y_train)**2)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 100 == 0 :
        print("Epoch {:4d}/{} | w1 {:.3f}, w2 {:.3f}, w3 {:.3f}, b {:.3f} | Cost {:.6f}".format(
            epoch, np_epochs, w1.item(), w2.item(), w3.item(), b.item(), cost.item()
        ))


# Epoch    0/2999 | w1 0.294, w2 0.294, w3 0.297, b 0.003 | Cost 29661.800781
# Epoch  100/2999 | w1 0.674, w2 0.661, w3 0.676, b 0.008 | Cost 1.563634
# Epoch  200/2999 | w1 0.679, w2 0.655, w3 0.677, b 0.008 | Cost 1.497607
# Epoch  300/2999 | w1 0.684, w2 0.649, w3 0.677, b 0.008 | Cost 1.435026
# Epoch  400/2999 | w1 0.689, w2 0.643, w3 0.678, b 0.008 | Cost 1.375730
# Epoch  500/2999 | w1 0.694, w2 0.638, w3 0.678, b 0.009 | Cost 1.319511
# Epoch  600/2999 | w1 0.699, w2 0.633, w3 0.679, b 0.009 | Cost 1.266222
# Epoch  700/2999 | w1 0.704, w2 0.627, w3 0.679, b 0.009 | Cost 1.215696
# Epoch  800/2999 | w1 0.709, w2 0.622, w3 0.679, b 0.009 | Cost 1.167818
# Epoch  900/2999 | w1 0.713, w2 0.617, w3 0.680, b 0.009 | Cost 1.122429
# Epoch 1000/2999 | w1 0.718, w2 0.613, w3 0.680, b 0.009 | Cost 1.079378
# Epoch 1100/2999 | w1 0.722, w2 0.608, w3 0.680, b 0.009 | Cost 1.038584
# Epoch 1200/2999 | w1 0.727, w2 0.603, w3 0.681, b 0.010 | Cost 0.999894
# Epoch 1300/2999 | w1 0.731, w2 0.599, w3 0.681, b 0.010 | Cost 0.963217
# Epoch 1400/2999 | w1 0.735, w2 0.595, w3 0.681, b 0.010 | Cost 0.928421
# Epoch 1500/2999 | w1 0.739, w2 0.591, w3 0.681, b 0.010 | Cost 0.895453
# Epoch 1600/2999 | w1 0.743, w2 0.586, w3 0.682, b 0.010 | Cost 0.864161
# Epoch 1700/2999 | w1 0.746, w2 0.583, w3 0.682, b 0.010 | Cost 0.834503
# Epoch 1800/2999 | w1 0.750, w2 0.579, w3 0.682, b 0.010 | Cost 0.806375
# Epoch 1900/2999 | w1 0.754, w2 0.575, w3 0.682, b 0.010 | Cost 0.779696
# Epoch 2000/2999 | w1 0.757, w2 0.571, w3 0.682, b 0.011 | Cost 0.754389
# Epoch 2100/2999 | w1 0.760, w2 0.568, w3 0.682, b 0.011 | Cost 0.730373
# Epoch 2200/2999 | w1 0.764, w2 0.564, w3 0.682, b 0.011 | Cost 0.707607
# Epoch 2300/2999 | w1 0.767, w2 0.561, w3 0.682, b 0.011 | Cost 0.685989
# Epoch 2400/2999 | w1 0.770, w2 0.558, w3 0.682, b 0.011 | Cost 0.665497
# Epoch 2500/2999 | w1 0.773, w2 0.555, w3 0.682, b 0.011 | Cost 0.646035
# Epoch 2600/2999 | w1 0.776, w2 0.552, w3 0.682, b 0.011 | Cost 0.627585
# Epoch 2700/2999 | w1 0.779, w2 0.549, w3 0.682, b 0.012 | Cost 0.610050
# Epoch 2800/2999 | w1 0.782, w2 0.546, w3 0.682, b 0.012 | Cost 0.593426
# Epoch 2900/2999 | w1 0.785, w2 0.543, w3 0.682, b 0.012 | Cost 0.577643


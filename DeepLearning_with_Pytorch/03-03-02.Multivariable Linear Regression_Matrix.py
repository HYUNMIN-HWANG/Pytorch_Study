# 다중 선형 회귀 : 다수의 x로부터 y를 예측
# 벡터와 행렬의 연산으로 바꾸기
# H(X) = XW + B (X:x데이터들로 이루어진 행렬, W:w가중치들로 이루어진 행렬, B:bias로 이루어진 벡터)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(2021)

'''Data'''
# 데이터 선언
# x1_train = torch.FloatTensor([[73], [93], [89], [96], [73]])
# x2_train = torch.FloatTensor([[80], [88], [91], [98], [66]])
# x3_train = torch.FloatTensor([[75], [93], [90], [100], [70]])
x_train = torch.FloatTensor([[73, 80, 75],
                             [93, 88, 93],
                             [89, 91, 80],
                             [96, 98, 100],
                             [73, 66, 70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])

print(x_train.shape)
print(y_train.shape)
# torch.Size([5, 3])
# torch.Size([5, 1])

# W, b 초기화
# w1 = torch.zeros(1, requires_grad=True)
# w2 = torch.zeros(1, requires_grad=True)
# w3 = torch.zeros(1, requires_grad=True)
w = torch.zeros((3,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# optimizer
# optimizer = optim.SGD([w1, w2, w3, b], lr=1e-5)
optimizer = optim.SGD([w, b], lr=1e-5)

np_epochs = 2999
for epoch in range(np_epochs + 1) :
    # 데이터 개수만큼 W를 곱해준다.
    # 단점 : 데이터의 개수가 많아지면 그만큼의 x와 w값을 정의해 줘야 한다. -> 비효울적임
    # hypothesis = x1_train * w1 + x2_train * w2 + x3_train * w3 + b
    hypothesis = x_train.matmul(w) + b

    cost = torch.mean((hypothesis-y_train)**2)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 100 == 0 :
        print("Epoch {:4d}/{} | hypothesis {} | Cost {:.6f}".format(
            epoch, np_epochs, hypothesis.squeeze().detach(), cost.item()
        ))

# Epoch    0/2999 | hypothesis tensor([0., 0., 0., 0., 0.]) | Cost 29661.800781
# Epoch  100/2999 | hypothesis tensor([154.0433, 185.0925, 175.8312, 198.5701, 141.2221]) | Cost 5.754573
# Epoch  200/2999 | hypothesis tensor([154.0278, 185.0649, 175.9335, 198.5128, 141.2284]) | Cost 5.512386
# Epoch  300/2999 | hypothesis tensor([154.0120, 185.0385, 176.0329, 198.4569, 141.2353]) | Cost 5.281667
# Epoch  400/2999 | hypothesis tensor([153.9960, 185.0133, 176.1295, 198.4022, 141.2426]) | Cost 5.061868
# Epoch  500/2999 | hypothesis tensor([153.9797, 184.9892, 176.2233, 198.3488, 141.2504]) | Cost 4.852424
# Epoch  600/2999 | hypothesis tensor([153.9632, 184.9662, 176.3143, 198.2966, 141.2586]) | Cost 4.652705
# Epoch  700/2999 | hypothesis tensor([153.9465, 184.9442, 176.4028, 198.2456, 141.2672]) | Cost 4.462287
# Epoch  800/2999 | hypothesis tensor([153.9296, 184.9232, 176.4888, 198.1958, 141.2762]) | Cost 4.280604
# Epoch  900/2999 | hypothesis tensor([153.9126, 184.9032, 176.5724, 198.1471, 141.2855]) | Cost 4.107294
# Epoch 1000/2999 | hypothesis tensor([153.8955, 184.8841, 176.6536, 198.0995, 141.2951]) | Cost 3.941866
# Epoch 1100/2999 | hypothesis tensor([153.8782, 184.8660, 176.7325, 198.0530, 141.3051]) | Cost 3.783911
# Epoch 1200/2999 | hypothesis tensor([153.8608, 184.8486, 176.8092, 198.0075, 141.3153]) | Cost 3.633077
# Epoch 1300/2999 | hypothesis tensor([153.8434, 184.8320, 176.8838, 197.9630, 141.3257]) | Cost 3.488997
# Epoch 1400/2999 | hypothesis tensor([153.8259, 184.8163, 176.9563, 197.9195, 141.3364]) | Cost 3.351316
# Epoch 1500/2999 | hypothesis tensor([153.8083, 184.8013, 177.0268, 197.8770, 141.3473]) | Cost 3.219756
# Epoch 1600/2999 | hypothesis tensor([153.7908, 184.7870, 177.0953, 197.8355, 141.3584]) | Cost 3.093989
# Epoch 1700/2999 | hypothesis tensor([153.7732, 184.7734, 177.1620, 197.7948, 141.3697]) | Cost 2.973708
# Epoch 1800/2999 | hypothesis tensor([153.7556, 184.7604, 177.2268, 197.7551, 141.3811]) | Cost 2.858705
# Epoch 1900/2999 | hypothesis tensor([153.7380, 184.7481, 177.2899, 197.7162, 141.3927]) | Cost 2.748643
# Epoch 2000/2999 | hypothesis tensor([153.7204, 184.7364, 177.3512, 197.6782, 141.4043]) | Cost 2.643338
# Epoch 2100/2999 | hypothesis tensor([153.7028, 184.7253, 177.4108, 197.6411, 141.4161]) | Cost 2.542579
# Epoch 2200/2999 | hypothesis tensor([153.6853, 184.7147, 177.4689, 197.6047, 141.4280]) | Cost 2.446103
# Epoch 2300/2999 | hypothesis tensor([153.6678, 184.7046, 177.5253, 197.5691, 141.4400]) | Cost 2.353729
# Epoch 2400/2999 | hypothesis tensor([153.6505, 184.6951, 177.5803, 197.5343, 141.4520]) | Cost 2.265264
# Epoch 2500/2999 | hypothesis tensor([153.6331, 184.6861, 177.6338, 197.5003, 141.4641]) | Cost 2.180532
# Epoch 2600/2999 | hypothesis tensor([153.6158, 184.6775, 177.6858, 197.4670, 141.4762]) | Cost 2.099355
# Epoch 2700/2999 | hypothesis tensor([153.5987, 184.6694, 177.7365, 197.4344, 141.4884]) | Cost 2.021556
# Epoch 2800/2999 | hypothesis tensor([153.5816, 184.6617, 177.7858, 197.4025, 141.5006]) | Cost 1.946995
# Epoch 2900/2999 | hypothesis tensor([153.5646, 184.6544, 177.8338, 197.3714, 141.5128]) | Cost 1.875525


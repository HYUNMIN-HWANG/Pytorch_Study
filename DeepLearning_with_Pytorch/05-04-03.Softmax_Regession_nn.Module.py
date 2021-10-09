# softmax regression 구현하기

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import optimizer 

torch.manual_seed(1)

'''DATA'''
x_train = [[1, 2, 1, 1],
           [2, 1, 3, 2],
           [3, 1, 3, 4],
           [4, 1, 5, 5],
           [1, 7, 5, 5],
           [1, 2, 5, 6],
           [1, 6, 6, 6],
           [1, 7, 7, 7]]    # 4개의 특성, 총 8개의 샘플
y_train = [2, 2, 2, 1, 1, 1, 0, 0]  # 라벨 0, 1, 2

x_train = torch.FloatTensor(x_train)
y_train = torch.LongTensor(y_train)

'''Softmax Regression (low-level)'''
print(x_train.shape)
print(y_train.shape)
# torch.Size([8, 4])
# torch.Size([8])

# one-hot encoding
y_one_hot = torch.zeros(8, 3)
y_one_hot.scatter_(1, y_train.unsqueeze(1), 1)
print(y_one_hot)
print(y_one_hot.shape)
# tensor([[0., 0., 1.],
#         [0., 0., 1.],
#         [0., 0., 1.],
#         [0., 1., 0.],
#         [0., 1., 0.],
#         [0., 1., 0.],
#         [1., 0., 0.],
#         [1., 0., 0.]])
# torch.Size([8, 3])

# 모델 초기화
# W= torch.zeros((4,3), requires_grad=True)
# b = torch.zeros(1, requires_grad=True)
model = nn.Linear(4, 3)

# optimizer = optim.SGD([W, b], lr=0.01)
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Train
nb_epochs = 1000
for epoch in range(nb_epochs+1):
    # hypothesis 필요없음
    # hypothesis = F.softmax(x_train.matmul(W)+b, dim=1)
    # z = x_train.matmul(W) + b
    prediction = model(x_train)

    # cost
    # cost = (y_one_hot * -torch.log(hypothesis)).sum(dim=1).mean()
    # cost = F.cross_entropy(z, y_train)
    cost = F.cross_entropy(prediction, y_train)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 100 == 0 :
        print("Epoch {:4d}/{} | Cost {:.6f}".format(
            epoch, nb_epochs, cost.item()
        ))

# Epoch    0/1000 | Cost 1.616785
# Epoch  100/1000 | Cost 0.863129
# Epoch  200/1000 | Cost 0.760346
# Epoch  300/1000 | Cost 0.707353
# Epoch  400/1000 | Cost 0.669059
# Epoch  500/1000 | Cost 0.639046
# Epoch  600/1000 | Cost 0.614619
# Epoch  700/1000 | Cost 0.594219
# Epoch  800/1000 | Cost 0.576826
# Epoch  900/1000 | Cost 0.561738
# Epoch 1000/1000 | Cost 0.548455

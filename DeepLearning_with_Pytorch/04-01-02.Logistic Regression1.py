import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(2021)

'''DATA'''
x_data = [[1,2],[2,3],[3,1],[4,3],[5,3],[6,2]]
y_data = [[0],[0],[0],[1],[1],[1]]
x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)

print(x_train.shape)
print(y_train.shape)
# torch.Size([6, 2])
# torch.Size([6, 1])

W = torch.zeros((2,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

'''Hypothesis'''

# 1. 직접 식 만들기
hypothesis = 1 / (1+torch.exp(-(x_train.matmul(W)+b)))
print(hypothesis)
# tensor([[0.5000], 
#         [0.5000], 
#         [0.5000],
#         [0.5000],
#         [0.5000],
#         [0.5000]], grad_fn=<MulBackward0>)  <-  초기값

# 2. torch.sigmoid
hypothesis = torch.sigmoid(x_train.matmul(W)+b)
print(hypothesis)
# tensor([[0.5000],
#         [0.5000],
#         [0.5000],
#         [0.5000],
#         [0.5000],
#         [0.5000]], grad_fn=<SigmoidBackward>)

'''Loss'''
# 1. 직접 식 만들기
losses = -(y_train * torch.log(hypothesis) + (1 - y_train) * torch.log(1 - hypothesis))
print(losses)
# tensor([[0.6931],
#         [0.6931],
#         [0.6931],
#         [0.6931],
#         [0.6931],
#         [0.6931]], grad_fn=<NegBackward>)

cost = losses.mean()
print(cost) # tensor(0.6931, grad_fn=<MeanBackward0>)

# 2. F.binary_cross_entropy
losses = F.binary_cross_entropy(hypothesis, y_train)
print(losses)   # tensor(0.6931, grad_fn=<BinaryCrossEntropyBackward>)


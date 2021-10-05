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

optimizer = optim.SGD([W, b], lr=1)

nb_epochs = 1000
for epoch in range(nb_epochs + 1) :
    '''Cost 계산'''
    hypothesis = torch.sigmoid(x_train.matmul(W)+b)
    cost = F.binary_cross_entropy(hypothesis, y_train)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 100 == 0 :
        print("Epoch {:4d}/{} | Cost {:.6f}".format(
            epoch, nb_epochs, cost.item()
        ))

# Epoch    0/1000 | Cost 0.693147
# Epoch  100/1000 | Cost 0.134722
# Epoch  200/1000 | Cost 0.080643
# Epoch  300/1000 | Cost 0.057900
# Epoch  400/1000 | Cost 0.045300
# Epoch  500/1000 | Cost 0.037261
# Epoch  600/1000 | Cost 0.031673
# Epoch  700/1000 | Cost 0.027556
# Epoch  800/1000 | Cost 0.024394
# Epoch  900/1000 | Cost 0.021888
# Epoch 1000/1000 | Cost 0.019852


'''Prediction'''
hypothesis = torch.sigmoid(x_train.matmul(W)+b)
print(hypothesis)
# tensor([[2.7648e-04],
#         [3.1608e-02],
#         [3.8977e-02],
#         [9.5622e-01],
#         [9.9823e-01],
#         [9.9969e-01]], grad_fn=<SigmoidBackward>)

# 0.5를 기준으로 넘으면 1, 안 넘으면 0
prediction = hypothesis >= torch.FloatTensor([0.5])
print(prediction)
# tensor([[False],
#         [False],
#         [False],
#         [ True],
#         [ True],
#         [ True]])

print(prediction.float())
# tensor([[0.],
#         [0.],
#         [0.],
#         [1.],
#         [1.],
#         [1.]])

'''훈련하고 난 뒤의 W, b'''
print("W : ", W)
print("b : ", b)
# W :  tensor([[3.2530],
#         [1.5179]], requires_grad=True)
# b :  tensor([-14.4819], requires_grad=True)

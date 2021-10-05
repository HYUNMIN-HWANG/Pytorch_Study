# nn.Module로 구현하는 로지스틱 회귀

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


# nn.Sequential() : nn.Moduel 층을 차례로 쌓을 수 있도록 한다.
model = nn.Sequential(
    nn.Linear(2,1), # input_dim = 2, output_dim = 1
    nn.Sigmoid()    # 시그모이드 함수를 거친다.
)

pred = model(x_train)
print(pred)
# tensor([[0.4646],
#         [0.3440],
#         [0.2304],
#         [0.1557],
#         [0.0986],
#         [0.0598]], grad_fn=<SigmoidBackward>)

'''optimizer'''
optimizer = optim.SGD(model.parameters(), lr=1)

nb_epochs = 1000
for epoch in range(nb_epochs + 1):
    hypothesis = model(x_train)

    cost = F.binary_cross_entropy(hypothesis, y_train)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()


    if epoch % 100 == 0 :
        prediction = hypothesis >= torch.FloatTensor([0.5]) # 0.5가 넘으면 True
        correct_prediction = prediction.float() == y_train  # y_train과 일치하면 True

        accuracy = correct_prediction.sum().item() / len(correct_prediction)

        print("Epoch {:4d}/{} | Cost {:.6f} Accuracy {:2.2f}%".format(
            epoch, nb_epochs, cost.item(), accuracy * 100
        ))

# Epoch    0/1000 | Cost 1.383666 Accuracy 50.00%
# Epoch  100/1000 | Cost 0.137943 Accuracy 100.00%
# Epoch  200/1000 | Cost 0.081756 Accuracy 100.00%
# Epoch  300/1000 | Cost 0.058462 Accuracy 100.00%
# Epoch  400/1000 | Cost 0.045639 Accuracy 100.00%
# Epoch  500/1000 | Cost 0.037489 Accuracy 100.00%
# Epoch  600/1000 | Cost 0.031836 Accuracy 100.00%
# Epoch  700/1000 | Cost 0.027679 Accuracy 100.00%
# Epoch  800/1000 | Cost 0.024491 Accuracy 100.00%
# Epoch  900/1000 | Cost 0.021966 Accuracy 100.00%
# Epoch 1000/1000 | Cost 0.019916 Accuracy 100.00%

'''Prediction'''
prediction = model(x_train)
print(prediction)
# tensor([[2.7866e-04],
#         [3.1704e-02],
#         [3.9103e-02],
#         [9.5609e-01],
#         [9.9822e-01],
#         [9.9969e-01]], grad_fn=<SigmoidBackward>) # 원래의 x_train과 똑같음

'''훈련 후의 W와 b값을 출력'''
print(list(model.parameters()))
# [Parameter containing:
# tensor([[3.2499, 1.5162]], requires_grad=True), Parameter containing:
# tensor([-14.4676], requires_grad=True)]


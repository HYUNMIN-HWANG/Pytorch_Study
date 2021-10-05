# 클래스로 파이토치 모델 구현하기 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.container import ModuleList
import torch.optim as optim

torch.manual_seed(2021)

'''DATA'''
x_data = [[1,2],[2,3],[3,1],[4,3],[5,3],[6,2]]
y_data = [[0],[0],[0],[1],[1],[1]]
x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)

'''Module'''
class BinaryClassifier(nn.Module):  # nn.Module을 상속받는다. 
    def __init__(self):
        super().__init__()  # super()를 사용(->nn.Modeule 클래스 속성들을 초기화한다.)해서 기반 클래스의 __init__ 메서드를 호출
        self.linear = nn.Linear(2,1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):   # model과 데이터를 함께 호출하면 자동으로 실행된다.
        return self.sigmoid(self.linear(x))

model = BinaryClassifier()


optimizer = optim.SGD(model.parameters(), lr=1)

nb_epochs = 1000
for epoch in range(nb_epochs+1):
    hypothesis = model(x_train)

    cost = F.binary_cross_entropy(hypothesis, y_train)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 100 == 0 :
        prediction = hypothesis >= torch.FloatTensor([0.5])
        correct_prediction = prediction.float() == y_train
        accuracy = correct_prediction.sum().item() / len(correct_prediction)
        print("Epoch {:4d}/{} | Cost {:.6f} | Accuracy {:2.2f}%".format(
            epoch, nb_epochs, cost.item(), accuracy*100
        ))

# Epoch    0/1000 | Cost 1.383666 | Accuracy 50.00%
# Epoch  100/1000 | Cost 0.137943 | Accuracy 100.00%
# Epoch  200/1000 | Cost 0.081756 | Accuracy 100.00%
# Epoch  300/1000 | Cost 0.058462 | Accuracy 100.00%
# Epoch  400/1000 | Cost 0.045639 | Accuracy 100.00%
# Epoch  500/1000 | Cost 0.037489 | Accuracy 100.00%
# Epoch  600/1000 | Cost 0.031836 | Accuracy 100.00%
# Epoch  700/1000 | Cost 0.027679 | Accuracy 100.00%
# Epoch  800/1000 | Cost 0.024491 | Accuracy 100.00%
# Epoch  900/1000 | Cost 0.021966 | Accuracy 100.00%
# Epoch 1000/1000 | Cost 0.019916 | Accuracy 100.00%

pred = model(x_train)
print(pred)
# tensor([[2.7866e-04],
#         [3.1704e-02],
#         [3.9103e-02],
#         [9.5609e-01],
#         [9.9822e-01],
#         [9.9969e-01]], grad_fn=<SigmoidBackward>)

print(list(model.parameters()))
# [Parameter containing:
# tensor([[3.2499, 1.5162]], requires_grad=True), Parameter containing:
# tensor([-14.4676], requires_grad=True)]
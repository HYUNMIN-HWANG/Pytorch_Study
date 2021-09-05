# Minibatch and Batch size

'''Minibatch & Batch size
- 전체 데이터를 더 작은 단위로 나누어서 해당 단위로 학습한다.
- 미니배치 만큼만 cost 계산하고 경사 하강법을 수행한다.
- 그 다음 미니배치 반복
- 전체 데이터에 대한 학습이 끝나면 1epoch이 끝나는 것
- 미니 배치의 크기 = batch size
- batch size는 주로 2의 제곱수를 사용한다.

- 배치 경사 하강법 : 전테 데이터에 대해서 한 번에 경사 하강법을 수행하는 방법 -> 계산량이 너무 많이 든다.
- 미니 배치 경사 하강법 : 미니 배치 단위로 경사 하강법을 수행하는 방법 -> 전체 데이터의 일부분만 보고 훈련하기 때문에 값이 수렴하는 데 안정적이지는 않지만, 훈련 속도가 빠르다.
'''

'''iteration
- 1 epoch 내에서 이루어지는 매개변수 W와 b의 업데이트 횟수
- eg) 전체 데이터가 2000이고 batch size가 200일 때 -> iteration=10 (1epch 당 10번의 업데이터가 이루어진다.)
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

torch.manual_seed(2021)

x_train = torch.FloatTensor([[73, 80, 75],
                             [93, 88, 93],
                             [89, 91, 80],
                             [96, 98, 100],
                             [73, 66, 70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])

# dataset으로 저장한다.
dataset = TensorDataset(x_train, y_train)

# DataLoad(데이터셋, 미니배치의 크기)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

model = nn.Linear(3,1)
optimizer = optim.SGD(model.parameters(), lr=1e-5)

nb_epochs=20
for epoch in range(nb_epochs+1):
    for batch_idx, samples in enumerate(dataloader):
        print(batch_idx)    # 0, 1, 2 : 3번 반복한다.
        print(samples)      # batch size 크기만큼 x값과 y값이 저장되어 있다.
        print("=======================")

        x_train, y_train = samples

        prediction = model(x_train)

        cost = F.mse_loss(prediction, y_train)

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        print("Epoch {:4d} / {} | Batch {} / {} | Cost {:.6f}".format(
            epoch, nb_epochs, batch_idx+1, len(dataloader), cost.item()
        ))

# predict
new_var = torch.FloatTensor([[73, 80, 75]])
pred_y = model(new_var)
print("훈련 후 입력이 73, 80, 75 일 때의 예측값 : ", pred_y)

# iteration 총 3번
# 0
# [tensor([[ 73.,  66.,  70.],
#         [ 96.,  98., 100.]]), tensor([[142.],
#         [196.]])]
# =======================
# Epoch    0 / 20 | Batch 1 / 3 | Cost 33058.718750
# 1
# [tensor([[73., 80., 75.],
#         [93., 88., 93.]]), tensor([[152.],
#         [185.]])]
# =======================
# Epoch    0 / 20 | Batch 2 / 3 | Cost 10373.256836
# 2
# [tensor([[89., 91., 80.]]), tensor([[180.]])]
# =======================

# Epoch    0 / 20 | Batch 1 / 3 | Cost 33058.718750
# Epoch    0 / 20 | Batch 2 / 3 | Cost 10373.256836
# Epoch    0 / 20 | Batch 3 / 3 | Cost 4775.863281
# Epoch    1 / 20 | Batch 1 / 3 | Cost 1001.335876
# Epoch    1 / 20 | Batch 2 / 3 | Cost 299.319366
# Epoch    1 / 20 | Batch 3 / 3 | Cost 88.523758
# Epoch    2 / 20 | Batch 1 / 3 | Cost 23.856028
# Epoch    2 / 20 | Batch 2 / 3 | Cost 62.486969
# Epoch    2 / 20 | Batch 3 / 3 | Cost 2.829404
# Epoch    3 / 20 | Batch 1 / 3 | Cost 8.837305
# Epoch    3 / 20 | Batch 2 / 3 | Cost 9.422228
# Epoch    3 / 20 | Batch 3 / 3 | Cost 61.302956
# Epoch    4 / 20 | Batch 1 / 3 | Cost 9.208483
# Epoch    4 / 20 | Batch 2 / 3 | Cost 40.493454
# Epoch    4 / 20 | Batch 3 / 3 | Cost 14.394021
# Epoch    5 / 20 | Batch 1 / 3 | Cost 7.990638
# Epoch    5 / 20 | Batch 2 / 3 | Cost 44.149048
# Epoch    5 / 20 | Batch 3 / 3 | Cost 2.768952
# Epoch    6 / 20 | Batch 1 / 3 | Cost 9.195355
# Epoch    6 / 20 | Batch 2 / 3 | Cost 10.481445
# Epoch    6 / 20 | Batch 3 / 3 | Cost 58.161804
# Epoch    7 / 20 | Batch 1 / 3 | Cost 19.912895
# Epoch    7 / 20 | Batch 2 / 3 | Cost 14.309592
# Epoch    7 / 20 | Batch 3 / 3 | Cost 49.869209
# Epoch    8 / 20 | Batch 1 / 3 | Cost 36.023651
# Epoch    8 / 20 | Batch 2 / 3 | Cost 14.476647
# Epoch    8 / 20 | Batch 3 / 3 | Cost 2.561045
# Epoch    9 / 20 | Batch 1 / 3 | Cost 31.217815
# Epoch    9 / 20 | Batch 2 / 3 | Cost 1.221712
# Epoch    9 / 20 | Batch 3 / 3 | Cost 17.086824
# Epoch   10 / 20 | Batch 1 / 3 | Cost 3.427038
# Epoch   10 / 20 | Batch 2 / 3 | Cost 29.044384
# Epoch   10 / 20 | Batch 3 / 3 | Cost 26.548847
# Epoch   11 / 20 | Batch 1 / 3 | Cost 42.738335
# Epoch   11 / 20 | Batch 2 / 3 | Cost 13.168743
# Epoch   11 / 20 | Batch 3 / 3 | Cost 7.192097
# Epoch   12 / 20 | Batch 1 / 3 | Cost 4.131483
# Epoch   12 / 20 | Batch 2 / 3 | Cost 39.246819
# Epoch   12 / 20 | Batch 3 / 3 | Cost 4.654087
# Epoch   13 / 20 | Batch 1 / 3 | Cost 8.271385
# Epoch   13 / 20 | Batch 2 / 3 | Cost 9.075732
# Epoch   13 / 20 | Batch 3 / 3 | Cost 61.659973
# Epoch   14 / 20 | Batch 1 / 3 | Cost 18.578730
# Epoch   14 / 20 | Batch 2 / 3 | Cost 18.961138
# Epoch   14 / 20 | Batch 3 / 3 | Cost 47.313053
# Epoch   15 / 20 | Batch 1 / 3 | Cost 3.390962
# Epoch   15 / 20 | Batch 2 / 3 | Cost 36.461544
# Epoch   15 / 20 | Batch 3 / 3 | Cost 4.356866
# Epoch   16 / 20 | Batch 1 / 3 | Cost 19.048143
# Epoch   16 / 20 | Batch 2 / 3 | Cost 38.287895
# Epoch   16 / 20 | Batch 3 / 3 | Cost 1.113557
# Epoch   17 / 20 | Batch 1 / 3 | Cost 20.942961
# Epoch   17 / 20 | Batch 2 / 3 | Cost 35.883392
# Epoch   17 / 20 | Batch 3 / 3 | Cost 0.318280
# Epoch   18 / 20 | Batch 1 / 3 | Cost 27.169636
# Epoch   18 / 20 | Batch 2 / 3 | Cost 20.316576
# Epoch   18 / 20 | Batch 3 / 3 | Cost 9.390686
# Epoch   19 / 20 | Batch 1 / 3 | Cost 35.332710
# Epoch   19 / 20 | Batch 2 / 3 | Cost 20.445408
# Epoch   19 / 20 | Batch 3 / 3 | Cost 9.015845
# Epoch   20 / 20 | Batch 1 / 3 | Cost 24.498209
# Epoch   20 / 20 | Batch 2 / 3 | Cost 16.942726
# Epoch   20 / 20 | Batch 3 / 3 | Cost 11.404843

# 훈련 후 입력이 73, 80, 75 일 때의 예측값 :  tensor([[154.2049]], grad_fn=<AddmmBackward>)


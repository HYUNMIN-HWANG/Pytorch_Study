# Batch Normalization
# Internal Covariance shift 현상을 방지하기 위함
# Internal Covariance shift : 각 layer마다 input 분포가 달라짐에 따라 학습 속도가 느려지는 현상
# Batch Normalization : Layer의 input 분포를 정규화해 학습 속도를 빠르게 하겠다.
# 학습 속도 향상, Gradient Vanishing 문제도 완화해줌

'''1. Module Import'''
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib import utils
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets

'''2. 딥러닝 모델을 설계할 떄 활용하는 장비 확인'''
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else :
    DEVICE = torch.device("cpu")

print(torch.__version__, DEVICE)    # 1.9.0 cuda

BATCH_SIZE = 32
EPOCHS = 10

'''3. MNIST 데이터 다운로드(train, test set 분리하기)'''
train_datasets = datasets.MNIST(
    root ='../data/MNIST',
    download=True,
    train=True,
    transform=transforms.ToTensor()
)

test_datasets = datasets.MNIST(
    root='../data/MNIST',
    train=False,
    transform=transforms.ToTensor()
)

train_loader = torch.utils.data.DataLoader(
    dataset=train_datasets,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    dataset=test_datasets,
    batch_size=BATCH_SIZE,
    shuffle=False
)

'''6. MLP(Multi Layer Perceptron) 모델 설계하기'''
class NET(nn.Module):
    def __init__(self):
        super(NET, self).__init__()
        self.fc1 = nn.Linear(28*28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
        self.dropout_prob = 0.5
        self.batch_norm1 = nn.BatchNorm1d(512)
        self.batch_norm2 = nn.BatchNorm1d(256)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.fc1(x)
        x = self.batch_norm1(x) # Batch Normalization 적용
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=self.dropout_prob)
        x = self.fc2(x)
        x = self.batch_norm2(x) # Batch Normalization 적용
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=self.dropout_prob)
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x

'''7. Optimizer, Objective Funtion'''
model = NET().to(DEVICE)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
criterion = nn.CrossEntropyLoss()

'''8. MLP 모델 학습을 진행하며 학습 데이터에 대한 모델 성능을 확인하는 함수 정의'''
def train(model, train_loader, optimizer, log_interval) :
    model.train()
    for batch_idx, (image, label) in enumerate(train_loader):
        image = image.to(DEVICE)
        label = label.to(DEVICE)
        optimizer.zero_grad()
        output = model(image)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0 :
            print("Train Epoch : {} [{}/{} ({:.0f}%)]\t Train Loss : {:.6f}".format(
                Epoch, batch_idx*len(image), len(train_loader.dataset), 100*batch_idx/len(train_loader), loss.item()
            ))

'''9. 학습되는 과정 속에서 검증 데이터에 대한 모델 성능을 확인하는 함수 정의'''
def evaluate(model, test_loader) :
    model.eval()
    test_loss=0
    correct=0

    with torch.no_grad() :
        for image, label in test_loader :
            image = image.to(DEVICE)
            label = label.to(DEVICE)
            output = model(image)
            test_loss += criterion(output, label).item()
            prediction = output.max(1, keepdim=True)[1]
            correct += prediction.eq(label.view_as(prediction)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    test_accuracy = 100 * correct / len(test_loader.dataset)
    return test_loss, test_accuracy

'''10. MLP 학습을 실행하면서 train, test set의 loss 및 test set accuracy 확인하기'''
for Epoch in range(1, EPOCHS+1) :
    train(model, train_loader, optimizer, log_interval=200)
    test_loss, test_accuracy = evaluate(model, test_loader)
    print("\n[Epoch : {}, \tTRAIN LOSS : {}\tTRAIN ACCURACY : {} %\n".format(
        Epoch, test_loss, test_accuracy
    ))

'''
Train Epoch : 1 [0/60000 (0%)]   Train Loss : 2.450984
Train Epoch : 1 [6400/60000 (11%)]       Train Loss : 0.610322
Train Epoch : 1 [12800/60000 (21%)]      Train Loss : 0.444699
Train Epoch : 1 [19200/60000 (32%)]      Train Loss : 0.512811
Train Epoch : 1 [25600/60000 (43%)]      Train Loss : 0.396239
Train Epoch : 1 [32000/60000 (53%)]      Train Loss : 0.336492
Train Epoch : 1 [38400/60000 (64%)]      Train Loss : 0.538761
Train Epoch : 1 [44800/60000 (75%)]      Train Loss : 0.228532
Train Epoch : 1 [51200/60000 (85%)]      Train Loss : 0.327542
Train Epoch : 1 [57600/60000 (96%)]      Train Loss : 0.214079

[Epoch : 1,     TRAIN LOSS : 0.004833098107669503       TRAIN ACCURACY : 95.19 %

Train Epoch : 2 [0/60000 (0%)]   Train Loss : 0.119046
Train Epoch : 2 [6400/60000 (11%)]       Train Loss : 0.101507
Train Epoch : 2 [12800/60000 (21%)]      Train Loss : 0.281230
Train Epoch : 2 [19200/60000 (32%)]      Train Loss : 0.353379
Train Epoch : 2 [25600/60000 (43%)]      Train Loss : 0.399651
Train Epoch : 2 [32000/60000 (53%)]      Train Loss : 0.147281
Train Epoch : 2 [38400/60000 (64%)]      Train Loss : 0.246596
Train Epoch : 2 [44800/60000 (75%)]      Train Loss : 0.207787
Train Epoch : 2 [51200/60000 (85%)]      Train Loss : 0.455577
Train Epoch : 2 [57600/60000 (96%)]      Train Loss : 0.136263

[Epoch : 2,     TRAIN LOSS : 0.003586506341502536       TRAIN ACCURACY : 96.48 %

Train Epoch : 3 [0/60000 (0%)]   Train Loss : 0.360477
Train Epoch : 3 [6400/60000 (11%)]       Train Loss : 0.174160
Train Epoch : 3 [12800/60000 (21%)]      Train Loss : 0.263171
Train Epoch : 3 [19200/60000 (32%)]      Train Loss : 0.252583
Train Epoch : 3 [25600/60000 (43%)]      Train Loss : 0.315485
Train Epoch : 3 [32000/60000 (53%)]      Train Loss : 0.378806
Train Epoch : 3 [38400/60000 (64%)]      Train Loss : 0.100037
Train Epoch : 3 [44800/60000 (75%)]      Train Loss : 0.246715
Train Epoch : 3 [51200/60000 (85%)]      Train Loss : 0.233912
Train Epoch : 3 [57600/60000 (96%)]      Train Loss : 0.061538

[Epoch : 3,     TRAIN LOSS : 0.00304706991854473        TRAIN ACCURACY : 97.06 %

Train Epoch : 4 [0/60000 (0%)]   Train Loss : 0.257343
Train Epoch : 4 [6400/60000 (11%)]       Train Loss : 0.017533
Train Epoch : 4 [12800/60000 (21%)]      Train Loss : 0.231534
Train Epoch : 4 [19200/60000 (32%)]      Train Loss : 0.368593
Train Epoch : 4 [25600/60000 (43%)]      Train Loss : 0.239482
Train Epoch : 4 [32000/60000 (53%)]      Train Loss : 0.126486
Train Epoch : 4 [38400/60000 (64%)]      Train Loss : 0.138451
Train Epoch : 4 [44800/60000 (75%)]      Train Loss : 0.061396
Train Epoch : 4 [51200/60000 (85%)]      Train Loss : 0.051254
Train Epoch : 4 [57600/60000 (96%)]      Train Loss : 0.483313

[Epoch : 4,     TRAIN LOSS : 0.002632473306468455       TRAIN ACCURACY : 97.47 %

Train Epoch : 5 [0/60000 (0%)]   Train Loss : 0.497468
Train Epoch : 5 [6400/60000 (11%)]       Train Loss : 0.304017
Train Epoch : 5 [12800/60000 (21%)]      Train Loss : 0.041925
Train Epoch : 5 [19200/60000 (32%)]      Train Loss : 0.172225
Train Epoch : 5 [25600/60000 (43%)]      Train Loss : 0.051897
Train Epoch : 5 [32000/60000 (53%)]      Train Loss : 0.155595
Train Epoch : 5 [38400/60000 (64%)]      Train Loss : 0.055731
Train Epoch : 5 [44800/60000 (75%)]      Train Loss : 0.078283
Train Epoch : 5 [51200/60000 (85%)]      Train Loss : 0.050622
Train Epoch : 5 [57600/60000 (96%)]      Train Loss : 0.286340

[Epoch : 5,     TRAIN LOSS : 0.00238618955303391        TRAIN ACCURACY : 97.79 %

Train Epoch : 6 [0/60000 (0%)]   Train Loss : 0.186605
Train Epoch : 6 [6400/60000 (11%)]       Train Loss : 0.192802
Train Epoch : 6 [12800/60000 (21%)]      Train Loss : 0.070616
Train Epoch : 6 [19200/60000 (32%)]      Train Loss : 0.279397
Train Epoch : 6 [25600/60000 (43%)]      Train Loss : 0.129911
Train Epoch : 6 [32000/60000 (53%)]      Train Loss : 0.100739
Train Epoch : 6 [38400/60000 (64%)]      Train Loss : 0.034150
Train Epoch : 6 [44800/60000 (75%)]      Train Loss : 0.237369
Train Epoch : 6 [51200/60000 (85%)]      Train Loss : 0.043697
Train Epoch : 6 [57600/60000 (96%)]      Train Loss : 0.142399

[Epoch : 6,     TRAIN LOSS : 0.0022978047736381996      TRAIN ACCURACY : 97.76 %

Train Epoch : 7 [0/60000 (0%)]   Train Loss : 0.060480
Train Epoch : 7 [6400/60000 (11%)]       Train Loss : 0.325182
Train Epoch : 7 [12800/60000 (21%)]      Train Loss : 0.098472
Train Epoch : 7 [19200/60000 (32%)]      Train Loss : 0.158841
Train Epoch : 7 [25600/60000 (43%)]      Train Loss : 0.032869
Train Epoch : 7 [32000/60000 (53%)]      Train Loss : 0.034357
Train Epoch : 7 [38400/60000 (64%)]      Train Loss : 0.253679
Train Epoch : 7 [44800/60000 (75%)]      Train Loss : 0.249772
Train Epoch : 7 [51200/60000 (85%)]      Train Loss : 0.146237
Train Epoch : 7 [57600/60000 (96%)]      Train Loss : 0.282822

[Epoch : 7,     TRAIN LOSS : 0.0021497314066597027      TRAIN ACCURACY : 97.79 %

Train Epoch : 8 [0/60000 (0%)]   Train Loss : 0.053153
Train Epoch : 8 [6400/60000 (11%)]       Train Loss : 0.201202
Train Epoch : 8 [12800/60000 (21%)]      Train Loss : 0.165956
Train Epoch : 8 [19200/60000 (32%)]      Train Loss : 0.205011
Train Epoch : 8 [25600/60000 (43%)]      Train Loss : 0.190476
Train Epoch : 8 [32000/60000 (53%)]      Train Loss : 0.145417
Train Epoch : 8 [38400/60000 (64%)]      Train Loss : 0.055138
Train Epoch : 8 [44800/60000 (75%)]      Train Loss : 0.054221
Train Epoch : 8 [51200/60000 (85%)]      Train Loss : 0.389874
Train Epoch : 8 [57600/60000 (96%)]      Train Loss : 0.230352

[Epoch : 8,     TRAIN LOSS : 0.0020792665665678215      TRAIN ACCURACY : 97.92 %

Train Epoch : 9 [0/60000 (0%)]   Train Loss : 0.119131
Train Epoch : 9 [6400/60000 (11%)]       Train Loss : 0.057063
Train Epoch : 9 [12800/60000 (21%)]      Train Loss : 0.018977
Train Epoch : 9 [19200/60000 (32%)]      Train Loss : 0.138926
Train Epoch : 9 [25600/60000 (43%)]      Train Loss : 0.217585
Train Epoch : 9 [32000/60000 (53%)]      Train Loss : 0.095302
Train Epoch : 9 [38400/60000 (64%)]      Train Loss : 0.096483
Train Epoch : 9 [44800/60000 (75%)]      Train Loss : 0.027000
Train Epoch : 9 [51200/60000 (85%)]      Train Loss : 0.058340
Train Epoch : 9 [57600/60000 (96%)]      Train Loss : 0.073643

[Epoch : 9,     TRAIN LOSS : 0.0019919423649909732      TRAIN ACCURACY : 98.02 %

Train Epoch : 10 [0/60000 (0%)]  Train Loss : 0.055472
Train Epoch : 10 [6400/60000 (11%)]      Train Loss : 0.125980
Train Epoch : 10 [12800/60000 (21%)]     Train Loss : 0.134159
Train Epoch : 10 [19200/60000 (32%)]     Train Loss : 0.335584
Train Epoch : 10 [25600/60000 (43%)]     Train Loss : 0.099057
Train Epoch : 10 [32000/60000 (53%)]     Train Loss : 0.054346
Train Epoch : 10 [38400/60000 (64%)]     Train Loss : 0.005818
Train Epoch : 10 [44800/60000 (75%)]     Train Loss : 0.086077
Train Epoch : 10 [51200/60000 (85%)]     Train Loss : 0.159762
Train Epoch : 10 [57600/60000 (96%)]     Train Loss : 0.082043

[Epoch : 10,    TRAIN LOSS : 0.0018327843277191277      TRAIN ACCURACY : 98.14 %
'''
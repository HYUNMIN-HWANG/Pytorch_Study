# Initialization
# 초기화 기법 :  신경망을 어떻게 초기화하느냐에 따라 학습 속도가 달라질 수 있다.
# 어디서부터 시작을 하느냐에 따라서 최적의 loss를 찾는 속도가 빨라질 수 있다.

# LeCun Initialization : LeCun Normal Initialization, LeCun Uniform Initialization
# He Initialization : Xavier Initialization을 relu 함수에 사용했을 때 비효율적이라는 것을 보완한 초기화 기법

'''1. Module Import'''
import numpy as np
import matplotlib.pyplot as plt
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

EPOCHS = 10
BATCH_SIZE = 32

'''3. MNIST 데이터 다운로드(train, test set 분리하기)'''
train_dataset = datasets.MNIST(
    root='../data/MNIST',
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

test_dataset = datasets.MNIST(
    root='../data/MNIST',
    train=False,
    transform=transforms.ToTensor()
)

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    shuffle=True,
    batch_size = BATCH_SIZE
)

test_loader = torch.utils.data.DataLoader(
    dataset = test_dataset,
    shuffle=False,
    batch_size = BATCH_SIZE

)

'''6. MLP(Multi Layer Perceptron) 모델 설계하기'''
class NET(nn.Module):
    def __init__(self) :
        super(NET, self).__init__()
        self.fc1 = nn.Linear(28*28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
        self.dropout_prob = 0.5
        self.batch_norm1 = nn.BatchNorm1d(512)
        self.batch_norm2 = nn.BatchNorm1d(256)
    
    def forward(self, x) :
        x = x.view(-1, 28*28)
        x = self.fc1(x)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=self.dropout_prob)
        x = self.fc2(x)
        x = self.batch_norm2(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=self.dropout_prob)
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x 

'''7. Optimizer, Objective Funtion'''
import torch.nn.init as init
def weight_init(m) :                            # weight 초기화
    if isinstance(m, nn.Linear) :               # nn.Linear에 해당하는 파라미터 값에 대해서만 저장
        init.kaiming_uniform_(m.weight.data)    # he_initialization을 이용해 파라미터 값 초기화
    
model = NET().to(DEVICE)
model.apply(weight_init)                        # 모델의 파라미터를 초기화한다.
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
criterion = nn.CrossEntropyLoss()

'''8. MLP 모델 학습을 진행하며 학습 데이터에 대한 모델 성능을 확인하는 함수 정의'''
def train(model, train_loader, optimizer, log_interval = 200 ):
    model.train()
    for batch_idx, (image, label) in enumerate(train_loader) :
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

    with torch.no_grad():
        for image, label in test_loader :
            image = image.to(DEVICE)
            label = label.to(DEVICE)
            output= model(image)
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
Train Epoch : 1 [0/60000 (0%)]   Train Loss : 3.144503
Train Epoch : 1 [6400/60000 (11%)]       Train Loss : 1.054975
Train Epoch : 1 [12800/60000 (21%)]      Train Loss : 0.636670
Train Epoch : 1 [19200/60000 (32%)]      Train Loss : 0.325025
Train Epoch : 1 [25600/60000 (43%)]      Train Loss : 0.490265
Train Epoch : 1 [32000/60000 (53%)]      Train Loss : 0.628072
Train Epoch : 1 [38400/60000 (64%)]      Train Loss : 0.483080
Train Epoch : 1 [44800/60000 (75%)]      Train Loss : 0.564792
Train Epoch : 1 [51200/60000 (85%)]      Train Loss : 0.215978
Train Epoch : 1 [57600/60000 (96%)]      Train Loss : 0.522428

[Epoch : 1,     TRAIN LOSS : 0.006969405706692487       TRAIN ACCURACY : 93.41 %

Train Epoch : 2 [0/60000 (0%)]   Train Loss : 0.423439
Train Epoch : 2 [6400/60000 (11%)]       Train Loss : 0.259258
Train Epoch : 2 [12800/60000 (21%)]      Train Loss : 0.353586
Train Epoch : 2 [19200/60000 (32%)]      Train Loss : 0.212406
Train Epoch : 2 [25600/60000 (43%)]      Train Loss : 0.249078
Train Epoch : 2 [32000/60000 (53%)]      Train Loss : 0.569144
Train Epoch : 2 [38400/60000 (64%)]      Train Loss : 0.556226
Train Epoch : 2 [44800/60000 (75%)]      Train Loss : 0.263603
Train Epoch : 2 [51200/60000 (85%)]      Train Loss : 0.356747
Train Epoch : 2 [57600/60000 (96%)]      Train Loss : 0.196788

[Epoch : 2,     TRAIN LOSS : 0.005548965122387744       TRAIN ACCURACY : 94.77 %

Train Epoch : 3 [0/60000 (0%)]   Train Loss : 0.507537
Train Epoch : 3 [6400/60000 (11%)]       Train Loss : 0.328070
Train Epoch : 3 [12800/60000 (21%)]      Train Loss : 0.200334
Train Epoch : 3 [19200/60000 (32%)]      Train Loss : 0.236503
Train Epoch : 3 [25600/60000 (43%)]      Train Loss : 0.136884
Train Epoch : 3 [32000/60000 (53%)]      Train Loss : 0.256345
Train Epoch : 3 [38400/60000 (64%)]      Train Loss : 0.295826
Train Epoch : 3 [44800/60000 (75%)]      Train Loss : 0.510915
Train Epoch : 3 [51200/60000 (85%)]      Train Loss : 0.156035
Train Epoch : 3 [57600/60000 (96%)]      Train Loss : 0.449028

[Epoch : 3,     TRAIN LOSS : 0.004719065459258855       TRAIN ACCURACY : 95.52 %

Train Epoch : 4 [0/60000 (0%)]   Train Loss : 0.201091
Train Epoch : 4 [6400/60000 (11%)]       Train Loss : 0.338349
Train Epoch : 4 [12800/60000 (21%)]      Train Loss : 0.443124
Train Epoch : 4 [19200/60000 (32%)]      Train Loss : 0.211451
Train Epoch : 4 [25600/60000 (43%)]      Train Loss : 0.202847
Train Epoch : 4 [32000/60000 (53%)]      Train Loss : 0.193337
Train Epoch : 4 [38400/60000 (64%)]      Train Loss : 0.163088
Train Epoch : 4 [44800/60000 (75%)]      Train Loss : 0.520339
Train Epoch : 4 [51200/60000 (85%)]      Train Loss : 0.107148
Train Epoch : 4 [57600/60000 (96%)]      Train Loss : 0.069578

[Epoch : 4,     TRAIN LOSS : 0.004089578925643582       TRAIN ACCURACY : 96.13 %

Train Epoch : 5 [0/60000 (0%)]   Train Loss : 0.146047
Train Epoch : 5 [6400/60000 (11%)]       Train Loss : 0.224916
Train Epoch : 5 [12800/60000 (21%)]      Train Loss : 0.148703
Train Epoch : 5 [19200/60000 (32%)]      Train Loss : 0.180407
Train Epoch : 5 [25600/60000 (43%)]      Train Loss : 0.483827
Train Epoch : 5 [32000/60000 (53%)]      Train Loss : 0.262639
Train Epoch : 5 [38400/60000 (64%)]      Train Loss : 0.459468
Train Epoch : 5 [44800/60000 (75%)]      Train Loss : 0.317421
Train Epoch : 5 [51200/60000 (85%)]      Train Loss : 0.248220
Train Epoch : 5 [57600/60000 (96%)]      Train Loss : 0.290481

[Epoch : 5,     TRAIN LOSS : 0.003758540361135965       TRAIN ACCURACY : 96.38 %

Train Epoch : 6 [0/60000 (0%)]   Train Loss : 0.408386
Train Epoch : 6 [6400/60000 (11%)]       Train Loss : 0.448098
Train Epoch : 6 [12800/60000 (21%)]      Train Loss : 0.135703
Train Epoch : 6 [19200/60000 (32%)]      Train Loss : 0.260828
Train Epoch : 6 [25600/60000 (43%)]      Train Loss : 0.231911
Train Epoch : 6 [32000/60000 (53%)]      Train Loss : 0.429191
Train Epoch : 6 [38400/60000 (64%)]      Train Loss : 0.565804
Train Epoch : 6 [44800/60000 (75%)]      Train Loss : 0.303989
Train Epoch : 6 [51200/60000 (85%)]      Train Loss : 0.167987
Train Epoch : 6 [57600/60000 (96%)]      Train Loss : 0.566528

[Epoch : 6,     TRAIN LOSS : 0.003493204381322721       TRAIN ACCURACY : 96.5 %

Train Epoch : 7 [0/60000 (0%)]   Train Loss : 0.139090
Train Epoch : 7 [6400/60000 (11%)]       Train Loss : 0.149685
Train Epoch : 7 [12800/60000 (21%)]      Train Loss : 0.189700
Train Epoch : 7 [19200/60000 (32%)]      Train Loss : 0.442192
Train Epoch : 7 [25600/60000 (43%)]      Train Loss : 0.251981
Train Epoch : 7 [32000/60000 (53%)]      Train Loss : 0.306839
Train Epoch : 7 [38400/60000 (64%)]      Train Loss : 0.362907
Train Epoch : 7 [44800/60000 (75%)]      Train Loss : 0.706821
Train Epoch : 7 [51200/60000 (85%)]      Train Loss : 0.149844
Train Epoch : 7 [57600/60000 (96%)]      Train Loss : 0.093512

[Epoch : 7,     TRAIN LOSS : 0.0033074918228492606      TRAIN ACCURACY : 96.76 %

Train Epoch : 8 [0/60000 (0%)]   Train Loss : 0.445516
Train Epoch : 8 [6400/60000 (11%)]       Train Loss : 0.070664
Train Epoch : 8 [12800/60000 (21%)]      Train Loss : 0.063043
Train Epoch : 8 [19200/60000 (32%)]      Train Loss : 0.140681
Train Epoch : 8 [25600/60000 (43%)]      Train Loss : 0.304215
Train Epoch : 8 [32000/60000 (53%)]      Train Loss : 0.383757
Train Epoch : 8 [38400/60000 (64%)]      Train Loss : 0.347681
Train Epoch : 8 [44800/60000 (75%)]      Train Loss : 0.178538
Train Epoch : 8 [51200/60000 (85%)]      Train Loss : 0.057008
Train Epoch : 8 [57600/60000 (96%)]      Train Loss : 0.453669

[Epoch : 8,     TRAIN LOSS : 0.0030885700364597143      TRAIN ACCURACY : 96.94 %

Train Epoch : 9 [0/60000 (0%)]   Train Loss : 0.065055
Train Epoch : 9 [6400/60000 (11%)]       Train Loss : 0.124615
Train Epoch : 9 [12800/60000 (21%)]      Train Loss : 0.138063
Train Epoch : 9 [19200/60000 (32%)]      Train Loss : 0.127433
Train Epoch : 9 [25600/60000 (43%)]      Train Loss : 0.205823
Train Epoch : 9 [32000/60000 (53%)]      Train Loss : 0.197434
Train Epoch : 9 [38400/60000 (64%)]      Train Loss : 0.374794
Train Epoch : 9 [44800/60000 (75%)]      Train Loss : 0.395469
Train Epoch : 9 [51200/60000 (85%)]      Train Loss : 0.242723
Train Epoch : 9 [57600/60000 (96%)]      Train Loss : 0.152192

[Epoch : 9,     TRAIN LOSS : 0.0029380349143029887      TRAIN ACCURACY : 97.15 %

Train Epoch : 10 [0/60000 (0%)]  Train Loss : 0.190907
Train Epoch : 10 [6400/60000 (11%)]      Train Loss : 0.051678
Train Epoch : 10 [12800/60000 (21%)]     Train Loss : 0.327202
Train Epoch : 10 [19200/60000 (32%)]     Train Loss : 0.078526
Train Epoch : 10 [25600/60000 (43%)]     Train Loss : 0.119187
Train Epoch : 10 [32000/60000 (53%)]     Train Loss : 0.180853
Train Epoch : 10 [38400/60000 (64%)]     Train Loss : 0.259762
Train Epoch : 10 [44800/60000 (75%)]     Train Loss : 0.349545
Train Epoch : 10 [51200/60000 (85%)]     Train Loss : 0.110430
Train Epoch : 10 [57600/60000 (96%)]     Train Loss : 0.010618

[Epoch : 10,    TRAIN LOSS : 0.0028271086052962346      TRAIN ACCURACY : 97.21 %
'''
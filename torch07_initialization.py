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
def train(model, train_loaderl, optimzer, log_interval = 200 ):
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
Train Epoch : 1 [0/60000 (0%)]   Train Loss : 3.071503
Train Epoch : 1 [6400/60000 (11%)]       Train Loss : 0.657135
Train Epoch : 1 [12800/60000 (21%)]      Train Loss : 0.792689
Train Epoch : 1 [19200/60000 (32%)]      Train Loss : 0.437815
Train Epoch : 1 [25600/60000 (43%)]      Train Loss : 0.592213
Train Epoch : 1 [32000/60000 (53%)]      Train Loss : 0.761029
Train Epoch : 1 [38400/60000 (64%)]      Train Loss : 0.509699
Train Epoch : 1 [44800/60000 (75%)]      Train Loss : 0.596523
Train Epoch : 1 [51200/60000 (85%)]      Train Loss : 0.519163
Train Epoch : 1 [57600/60000 (96%)]      Train Loss : 0.516142

[Epoch : 1,     TRAIN LOSS : 0.006906039650691673       TRAIN ACCURACY : 93.38 %

Train Epoch : 2 [0/60000 (0%)]   Train Loss : 0.455080
Train Epoch : 2 [6400/60000 (11%)]       Train Loss : 0.400566
Train Epoch : 2 [12800/60000 (21%)]      Train Loss : 0.740581
Train Epoch : 2 [19200/60000 (32%)]      Train Loss : 0.215031
Train Epoch : 2 [25600/60000 (43%)]      Train Loss : 0.210111
Train Epoch : 2 [32000/60000 (53%)]      Train Loss : 0.299595
Train Epoch : 2 [38400/60000 (64%)]      Train Loss : 0.277303
Train Epoch : 2 [44800/60000 (75%)]      Train Loss : 0.500054
Train Epoch : 2 [51200/60000 (85%)]      Train Loss : 0.425607
Train Epoch : 2 [57600/60000 (96%)]      Train Loss : 0.693577

[Epoch : 2,     TRAIN LOSS : 0.005235680847731419       TRAIN ACCURACY : 95.0 %

Train Epoch : 3 [0/60000 (0%)]   Train Loss : 0.333362
Train Epoch : 3 [6400/60000 (11%)]       Train Loss : 0.346759
Train Epoch : 3 [12800/60000 (21%)]      Train Loss : 0.190206
Train Epoch : 3 [19200/60000 (32%)]      Train Loss : 0.111698
Train Epoch : 3 [25600/60000 (43%)]      Train Loss : 0.229040
Train Epoch : 3 [32000/60000 (53%)]      Train Loss : 0.180761
Train Epoch : 3 [38400/60000 (64%)]      Train Loss : 0.308326
Train Epoch : 3 [44800/60000 (75%)]      Train Loss : 0.243012
Train Epoch : 3 [51200/60000 (85%)]      Train Loss : 0.344783
Train Epoch : 3 [57600/60000 (96%)]      Train Loss : 0.158371

[Epoch : 3,     TRAIN LOSS : 0.004478329095302615       TRAIN ACCURACY : 95.67 %

Train Epoch : 4 [0/60000 (0%)]   Train Loss : 0.482504
Train Epoch : 4 [6400/60000 (11%)]       Train Loss : 0.596754
Train Epoch : 4 [12800/60000 (21%)]      Train Loss : 0.198388
Train Epoch : 4 [19200/60000 (32%)]      Train Loss : 0.658182
Train Epoch : 4 [25600/60000 (43%)]      Train Loss : 0.254438
Train Epoch : 4 [32000/60000 (53%)]      Train Loss : 0.299325
Train Epoch : 4 [38400/60000 (64%)]      Train Loss : 0.283126
Train Epoch : 4 [44800/60000 (75%)]      Train Loss : 0.386536
Train Epoch : 4 [51200/60000 (85%)]      Train Loss : 0.237743
Train Epoch : 4 [57600/60000 (96%)]      Train Loss : 0.296918

[Epoch : 4,     TRAIN LOSS : 0.004115665020735469       TRAIN ACCURACY : 96.02 %

Train Epoch : 5 [0/60000 (0%)]   Train Loss : 0.415785
Train Epoch : 5 [6400/60000 (11%)]       Train Loss : 0.184928
Train Epoch : 5 [12800/60000 (21%)]      Train Loss : 0.125872
Train Epoch : 5 [19200/60000 (32%)]      Train Loss : 0.296216
Train Epoch : 5 [25600/60000 (43%)]      Train Loss : 0.237389
Train Epoch : 5 [32000/60000 (53%)]      Train Loss : 0.286602
Train Epoch : 5 [38400/60000 (64%)]      Train Loss : 0.293726
Train Epoch : 5 [44800/60000 (75%)]      Train Loss : 0.112850
Train Epoch : 5 [51200/60000 (85%)]      Train Loss : 0.419207
Train Epoch : 5 [57600/60000 (96%)]      Train Loss : 0.309397

[Epoch : 5,     TRAIN LOSS : 0.0037596526029345114      TRAIN ACCURACY : 96.33 %

Train Epoch : 6 [0/60000 (0%)]   Train Loss : 0.212452
Train Epoch : 6 [6400/60000 (11%)]       Train Loss : 0.182263
Train Epoch : 6 [12800/60000 (21%)]      Train Loss : 0.102841
Train Epoch : 6 [19200/60000 (32%)]      Train Loss : 0.087606
Train Epoch : 6 [25600/60000 (43%)]      Train Loss : 0.141872
Train Epoch : 6 [32000/60000 (53%)]      Train Loss : 0.099373
Train Epoch : 6 [38400/60000 (64%)]      Train Loss : 0.260665
Train Epoch : 6 [44800/60000 (75%)]      Train Loss : 0.231244
Train Epoch : 6 [51200/60000 (85%)]      Train Loss : 0.158571
Train Epoch : 6 [57600/60000 (96%)]      Train Loss : 0.107871

[Epoch : 6,     TRAIN LOSS : 0.003399082719249418       TRAIN ACCURACY : 96.62 %

Train Epoch : 7 [0/60000 (0%)]   Train Loss : 0.174941
Train Epoch : 7 [6400/60000 (11%)]       Train Loss : 0.106487
Train Epoch : 7 [12800/60000 (21%)]      Train Loss : 0.122129
Train Epoch : 7 [19200/60000 (32%)]      Train Loss : 0.277171
Train Epoch : 7 [25600/60000 (43%)]      Train Loss : 0.112076
Train Epoch : 7 [32000/60000 (53%)]      Train Loss : 0.159021
Train Epoch : 7 [38400/60000 (64%)]      Train Loss : 0.155808
Train Epoch : 7 [44800/60000 (75%)]      Train Loss : 0.071905
Train Epoch : 7 [51200/60000 (85%)]      Train Loss : 0.311691
Train Epoch : 7 [57600/60000 (96%)]      Train Loss : 0.136790

[Epoch : 7,     TRAIN LOSS : 0.003214670426107477       TRAIN ACCURACY : 96.82 %

Train Epoch : 8 [0/60000 (0%)]   Train Loss : 0.202697
Train Epoch : 8 [6400/60000 (11%)]       Train Loss : 0.405036
Train Epoch : 8 [12800/60000 (21%)]      Train Loss : 0.160187
Train Epoch : 8 [19200/60000 (32%)]      Train Loss : 0.097893
Train Epoch : 8 [25600/60000 (43%)]      Train Loss : 0.235711
Train Epoch : 8 [32000/60000 (53%)]      Train Loss : 0.253371
Train Epoch : 8 [38400/60000 (64%)]      Train Loss : 0.249992
Train Epoch : 8 [44800/60000 (75%)]      Train Loss : 0.179842
Train Epoch : 8 [51200/60000 (85%)]      Train Loss : 0.238449
Train Epoch : 8 [57600/60000 (96%)]      Train Loss : 0.355336

[Epoch : 8,     TRAIN LOSS : 0.00304179914080305        TRAIN ACCURACY : 97.05 %

Train Epoch : 9 [0/60000 (0%)]   Train Loss : 0.181973
Train Epoch : 9 [6400/60000 (11%)]       Train Loss : 0.529945
Train Epoch : 9 [12800/60000 (21%)]      Train Loss : 0.172534
Train Epoch : 9 [19200/60000 (32%)]      Train Loss : 0.203331
Train Epoch : 9 [25600/60000 (43%)]      Train Loss : 0.239420
Train Epoch : 9 [32000/60000 (53%)]      Train Loss : 0.120711
Train Epoch : 9 [38400/60000 (64%)]      Train Loss : 0.065951
Train Epoch : 9 [44800/60000 (75%)]      Train Loss : 0.315071
Train Epoch : 9 [51200/60000 (85%)]      Train Loss : 0.057020
Train Epoch : 9 [57600/60000 (96%)]      Train Loss : 0.186332

[Epoch : 9,     TRAIN LOSS : 0.0030169738753174895      TRAIN ACCURACY : 96.98 %

Train Epoch : 10 [0/60000 (0%)]  Train Loss : 0.052472
Train Epoch : 10 [6400/60000 (11%)]      Train Loss : 0.231316
Train Epoch : 10 [12800/60000 (21%)]     Train Loss : 0.151695
Train Epoch : 10 [19200/60000 (32%)]     Train Loss : 0.236031
Train Epoch : 10 [25600/60000 (43%)]     Train Loss : 0.157391
Train Epoch : 10 [32000/60000 (53%)]     Train Loss : 0.143079
Train Epoch : 10 [38400/60000 (64%)]     Train Loss : 0.237426
Train Epoch : 10 [44800/60000 (75%)]     Train Loss : 0.133948
Train Epoch : 10 [51200/60000 (85%)]     Train Loss : 0.128911
Train Epoch : 10 [57600/60000 (96%)]     Train Loss : 0.049402

[Epoch : 10,    TRAIN LOSS : 0.0028393972476682392      TRAIN ACCURACY : 97.24 %
'''
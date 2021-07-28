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
    DEVICE = torch.devide("cpu")

print(torch.__version__, DEVICE)    # 1.9.0 cuda

BATCH_SIZE = 32
EPOCHS = 10

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

train_loader=torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader=torch.utils.data.DataLoader(
    dataset=test_dataset,
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
    
    def forward(self, x) :
        x = x.view(-1, 28*28)
        x = self.fc1(x)
        x = F.relu(x)   
        # ReLU : 입력 값이 0이상이면 그대로 출력하고, 0이하면 0으로 출력하는 함수
        # 미분할 때, 입력 값이 0이상인 부분은 기울기가 1, 0이하인 부분은 기울기가 0
        # Bach Propagation 하면 아예 없어지거나 완전히 살리거나
        # Gradient Vanishing이 일어나는 것을 완화시킬 수 있다. 
        x = F.dropout(x, training=self.training, p=self.dropout_prob)
        x = self.fc2(x)
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
            print("Train Epoch : {} [{}/{}({:.0f}%)]\tTrain loss : {:.6f}".format(
                Epoch, batch_idx*len(image), len(train_loader.dataset), 100*batch_idx/len(train_loader), loss.item()
            ))

'''9. 학습되는 과정 속에서 검증 데이터에 대한 모델 성능을 확인하는 함수 정의'''
def evaluate(model, test_loader) :
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for image, label in test_loader :
            image = image.to(DEVICE)
            label = label.to(DEVICE)
            output = model(image)
            test_loss += criterion(output, label).item()
            prediction = output.max(1, keepdim=True)[1]
            correct += prediction.eq(label.view_as(prediction)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    test_accuracy = 100*correct / len(test_loader.dataset)
    return test_loss, test_accuracy

'''10. MLP 학습을 실행하면서 train, test set의 loss 및 test set accuracy 확인하기'''
for Epoch in range(1, EPOCHS+1) :
    train(model, train_loader, optimizer, log_interval=200)
    test_loss, test_accuracy = evaluate(model, test_loader)
    print("\n[Epoch : {},\tTEST LOSS : {:.4f},\tTEST ACCURACY : {:.2f} %\n".format(
        Epoch, test_loss, test_accuracy
    ))

'''Train Epoch : 1 [0/60000(0%)]   Train loss : 2.312081
Train Epoch : 1 [6400/60000(11%)]       Train loss : 2.077765
Train Epoch : 1 [12800/60000(21%)]      Train loss : 1.434386
Train Epoch : 1 [19200/60000(32%)]      Train loss : 0.790514
Train Epoch : 1 [25600/60000(43%)]      Train loss : 0.616529
Train Epoch : 1 [32000/60000(53%)]      Train loss : 0.610253
Train Epoch : 1 [38400/60000(64%)]      Train loss : 0.483479
Train Epoch : 1 [44800/60000(75%)]      Train loss : 0.381728
Train Epoch : 1 [51200/60000(85%)]      Train loss : 0.397402
Train Epoch : 1 [57600/60000(96%)]      Train loss : 0.648905

[Epoch : 1,     TEST LOSS : 0.0100,     TEST ACCURACY : 90.92 %

Train Epoch : 2 [0/60000(0%)]   Train loss : 0.385808
Train Epoch : 2 [6400/60000(11%)]       Train loss : 0.423779
Train Epoch : 2 [12800/60000(21%)]      Train loss : 0.808531
Train Epoch : 2 [19200/60000(32%)]      Train loss : 0.240419
Train Epoch : 2 [25600/60000(43%)]      Train loss : 0.201089
Train Epoch : 2 [32000/60000(53%)]      Train loss : 0.439162
Train Epoch : 2 [38400/60000(64%)]      Train loss : 0.359069
Train Epoch : 2 [44800/60000(75%)]      Train loss : 0.300860
Train Epoch : 2 [51200/60000(85%)]      Train loss : 0.468393
Train Epoch : 2 [57600/60000(96%)]      Train loss : 0.145870

[Epoch : 2,     TEST LOSS : 0.0070,     TEST ACCURACY : 93.41 %

Train Epoch : 3 [0/60000(0%)]   Train loss : 0.275349
Train Epoch : 3 [6400/60000(11%)]       Train loss : 0.179139
Train Epoch : 3 [12800/60000(21%)]      Train loss : 0.261233
Train Epoch : 3 [19200/60000(32%)]      Train loss : 0.535027
Train Epoch : 3 [25600/60000(43%)]      Train loss : 0.131906
Train Epoch : 3 [32000/60000(53%)]      Train loss : 0.103930
Train Epoch : 3 [38400/60000(64%)]      Train loss : 0.115245
Train Epoch : 3 [44800/60000(75%)]      Train loss : 0.108581
Train Epoch : 3 [51200/60000(85%)]      Train loss : 0.243900
Train Epoch : 3 [57600/60000(96%)]      Train loss : 0.207686

[Epoch : 3,     TEST LOSS : 0.0055,     TEST ACCURACY : 94.67 %

Train Epoch : 4 [0/60000(0%)]   Train loss : 0.568885
Train Epoch : 4 [6400/60000(11%)]       Train loss : 0.118232
Train Epoch : 4 [12800/60000(21%)]      Train loss : 0.358127
Train Epoch : 4 [19200/60000(32%)]      Train loss : 0.174060
Train Epoch : 4 [25600/60000(43%)]      Train loss : 0.309348
Train Epoch : 4 [32000/60000(53%)]      Train loss : 0.121012
Train Epoch : 4 [38400/60000(64%)]      Train loss : 0.077179
Train Epoch : 4 [44800/60000(75%)]      Train loss : 0.121806
Train Epoch : 4 [51200/60000(85%)]      Train loss : 0.051003
Train Epoch : 4 [57600/60000(96%)]      Train loss : 0.154544

[Epoch : 4,     TEST LOSS : 0.0045,     TEST ACCURACY : 95.65 %

Train Epoch : 5 [0/60000(0%)]   Train loss : 0.198351
Train Epoch : 5 [6400/60000(11%)]       Train loss : 0.200660
Train Epoch : 5 [12800/60000(21%)]      Train loss : 0.024739
Train Epoch : 5 [19200/60000(32%)]      Train loss : 0.129581
Train Epoch : 5 [25600/60000(43%)]      Train loss : 0.324852
Train Epoch : 5 [32000/60000(53%)]      Train loss : 0.375477
Train Epoch : 5 [38400/60000(64%)]      Train loss : 0.079885
Train Epoch : 5 [44800/60000(75%)]      Train loss : 0.574140
Train Epoch : 5 [51200/60000(85%)]      Train loss : 0.186359
Train Epoch : 5 [57600/60000(96%)]      Train loss : 0.185188

[Epoch : 5,     TEST LOSS : 0.0039,     TEST ACCURACY : 96.22 %

Train Epoch : 6 [0/60000(0%)]   Train loss : 0.189129
Train Epoch : 6 [6400/60000(11%)]       Train loss : 0.066491
Train Epoch : 6 [12800/60000(21%)]      Train loss : 0.164968
Train Epoch : 6 [19200/60000(32%)]      Train loss : 0.459517
Train Epoch : 6 [25600/60000(43%)]      Train loss : 0.079466
Train Epoch : 6 [32000/60000(53%)]      Train loss : 0.115543
Train Epoch : 6 [38400/60000(64%)]      Train loss : 0.064854
Train Epoch : 6 [44800/60000(75%)]      Train loss : 0.149205
Train Epoch : 6 [51200/60000(85%)]      Train loss : 0.263682
Train Epoch : 6 [57600/60000(96%)]      Train loss : 0.095215

[Epoch : 6,     TEST LOSS : 0.0034,     TEST ACCURACY : 96.65 %

Train Epoch : 7 [0/60000(0%)]   Train loss : 0.757092
Train Epoch : 7 [6400/60000(11%)]       Train loss : 0.163891
Train Epoch : 7 [12800/60000(21%)]      Train loss : 0.340748
Train Epoch : 7 [19200/60000(32%)]      Train loss : 0.363667
Train Epoch : 7 [25600/60000(43%)]      Train loss : 0.042316
Train Epoch : 7 [32000/60000(53%)]      Train loss : 0.178792
Train Epoch : 7 [38400/60000(64%)]      Train loss : 0.044834
Train Epoch : 7 [44800/60000(75%)]      Train loss : 0.054349
Train Epoch : 7 [51200/60000(85%)]      Train loss : 0.252885
Train Epoch : 7 [57600/60000(96%)]      Train loss : 0.111771

[Epoch : 7,     TEST LOSS : 0.0032,     TEST ACCURACY : 96.87 %

Train Epoch : 8 [0/60000(0%)]   Train loss : 0.335326
Train Epoch : 8 [6400/60000(11%)]       Train loss : 0.237446
Train Epoch : 8 [12800/60000(21%)]      Train loss : 0.154115
Train Epoch : 8 [19200/60000(32%)]      Train loss : 0.486720
Train Epoch : 8 [25600/60000(43%)]      Train loss : 0.072281
Train Epoch : 8 [32000/60000(53%)]      Train loss : 0.117852
Train Epoch : 8 [38400/60000(64%)]      Train loss : 0.252164
Train Epoch : 8 [44800/60000(75%)]      Train loss : 0.096099
Train Epoch : 8 [51200/60000(85%)]      Train loss : 0.054045
Train Epoch : 8 [57600/60000(96%)]      Train loss : 0.140615

[Epoch : 8,     TEST LOSS : 0.0030,     TEST ACCURACY : 97.08 %

Train Epoch : 9 [0/60000(0%)]   Train loss : 0.044683
Train Epoch : 9 [6400/60000(11%)]       Train loss : 0.212025
Train Epoch : 9 [12800/60000(21%)]      Train loss : 0.081165
Train Epoch : 9 [19200/60000(32%)]      Train loss : 0.243447
Train Epoch : 9 [25600/60000(43%)]      Train loss : 0.030431
Train Epoch : 9 [32000/60000(53%)]      Train loss : 0.213503
Train Epoch : 9 [38400/60000(64%)]      Train loss : 0.105595
Train Epoch : 9 [44800/60000(75%)]      Train loss : 0.156062
Train Epoch : 9 [51200/60000(85%)]      Train loss : 0.066919
Train Epoch : 9 [57600/60000(96%)]      Train loss : 0.228082

[Epoch : 9,     TEST LOSS : 0.0028,     TEST ACCURACY : 97.31 %

Train Epoch : 10 [0/60000(0%)]  Train loss : 0.170634
Train Epoch : 10 [6400/60000(11%)]      Train loss : 0.212103
Train Epoch : 10 [12800/60000(21%)]     Train loss : 0.132469
Train Epoch : 10 [19200/60000(32%)]     Train loss : 0.021007
Train Epoch : 10 [25600/60000(43%)]     Train loss : 0.019907
Train Epoch : 10 [32000/60000(53%)]     Train loss : 0.242015
Train Epoch : 10 [38400/60000(64%)]     Train loss : 0.027846
Train Epoch : 10 [44800/60000(75%)]     Train loss : 0.247815
Train Epoch : 10 [51200/60000(85%)]     Train loss : 0.319070
Train Epoch : 10 [57600/60000(96%)]     Train loss : 0.132556

[Epoch : 10,    TEST LOSS : 0.0027,     TEST ACCURACY : 97.36 %
'''
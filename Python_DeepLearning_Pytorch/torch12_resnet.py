# ResNet

'''1. Module Import'''
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets

'''2. 딥러닝 모델을 설계할 때 활용하는 장비 확인'''
if torch.cuda.is_available() :
    DEVICE = torch.device("cuda")
else :
    DEVICE = torch.device("cpu")

print(torch.__version__, DEVICE)    # 1.9.0 cuda

BATCH_SIZE = 32
EPOCHS = 10

'''3. CIFAR10 데이터 다운로드 (train, test set 분리하기)'''
train_dataset=datasets.CIFAR10(
    root="../data/CIFAR_10",
    train=True,
    download=True,
    transform=transforms.Compose([                              # 이미지 데이터에 전처리 및 Augmentation을 다양하게 적용할 때 이용하는 메서드
        transforms.RandomHorizontalFlip(),                      # 50% 확률로 좌우반전
        transforms.ToTensor(),                                  # 이미제 데이터를 0과 1사이의 값으로 정규화
        transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))    #(평균:red,green,blue 순으로 0.5씩 적용),(표준편차:red, green, blue 순으로 0.5씩 적용)
    ])
)

test_dataset=datasets.CIFAR10(                              # 기본적으로 학습 데이터에 이용하는 전처리 과정은 검증 데이터에도 동일하게 적용돼야 모델의 성능을 평가할 수 있습니다.
    root="../data/CIFAR_10",
    train=False,
    transform=transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
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


'''4. 데이터 확인하기 (1)'''
for (X_train, y_train) in train_loader :
    print("X_train : ", X_train.size(), "type : ", X_train.type())
    print("y_train : ", y_train.size(), "type : ", y_train.type())

# X_train :  torch.Size([32, 3, 32, 32]) type :  torch.FloatTensor
# y_train :  torch.Size([32]) type :  torch.LongTensor

'''5. 데이터 확인하기 (2)'''
pltsize=1
plt.figure(figsize=(10*pltsize, pltsize))

for i in range(10) :
    plt.subplot(1, 10, i+1)
    plt.axis('off')
    plt.imshow(np.transpose(X_train[i], (1,2,0)))
    # [Batch Size, Channel, Width, Height] -> [Width, Height, Channel] 형태로 바꿔준다.
    plt.title("Class : " + str(y_train[i].item()))
plt.show()

'''6. ResNet 모델 설계하기'''
class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes,   # in_planes : 데이터의 채널 수, planes : 필터 수 
            kernel_size=3,                          # filter 3*3 크기로 설정
            stride=stride,                          # stride 수 만큼 이동
            padding=1, 
            bias=False)                             # 연산을 한 이후 Bias 값을 더해줄 것인지를 선택하는 옵션
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()         # shorcut 정의 : 기존의 값과 Convolution,Batch Normalization 한 결과를 더하는 과정
        if stride != 1 or in_planes != planes:  # 두 번째 블록부터 적용되는 shortcut을 정의
            self.shortcut = nn.Sequential(      
                nn.Conv2d(in_planes, planes, 
                    kernel_size=1,
                    stride=stride,
                    bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x) :
        out = F.relu(self.bn1(self.conv1(x)))   # conv1 -> bn1 -> relu
        out = self.bn2(self.conv2(out))         # conv2 -> bn2
        out += self.shortcut(x)                 # shortcut의 결과와 out의 결과를 합쳐준다.
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, num_classes=10):         # 클래스 개수 10개
        super(ResNet, self).__init__()
        self.in_planes=16
        self.conv1=nn.Conv2d(3, 16, 
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False)
        self.bn1=nn.BatchNorm2d(16)
        self.layer1=self._make_layer(16, 2, stride=1)
        self.layer2=self._make_layer(32, 2, stride=2)
        self.layer3=self._make_layer(64, 2, stride=2)
        self.linear=nn.Linear(64, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks -1)  # stride 범위를 각 BasicBlock마다 다르게 구성할 수 있다.
        print("****************** strides : ", strides) # [1, 1] || [2, 1] || [2, 1]
        layers = []                                 # BasicBlock을 통해 생성된 결괏값을 추가하기 위해 빈 리스트를 정의함
        for stride in strides:
            print("================= stride : ", stride)    # 1, 1 || 2, 1 || 2, 1
            layers.append(BasicBlock(self.in_planes, planes, stride))   # BasicBlock 결과값을 layer 리스트에 추가한다.
            self.in_planes = planes
        return nn.Sequential(*layers)   # 여러 층으로 생성한 레이러를 Sequential 내에 정의
    
    def forward(self, x) :
        out = F.relu(self.bn1(self.conv1(x)))   # conv1 -> bn1 -> relu
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8)              # 8*8 크기의 filter
        out = out.view(out.size(0), -1)         # 1차원의 벡터로 펼쳐줌
        out = self.linear(out)
        return out


'''7. Optimizer, Objective Function 설정하기'''
model = ResNet().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

print(model)
# ResNet(
#   (conv1): Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#   (bn1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (layer1): Sequential(
#     (0): BasicBlock(
#       (conv1): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), bias=False)
#       (bn2): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (shortcut): Sequential()
#     )
#     (1): BasicBlock(
#       (conv1): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), bias=False)
#       (bn2): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (shortcut): Sequential()
#     )
#   )
#   (layer2): Sequential(
#     (0): BasicBlock(
#       (conv1): Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
#       (bn1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), bias=False)
#       (bn2): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (shortcut): Sequential(
#         (0): Conv2d(16, 32, kernel_size=(1, 1), stride=(2, 2), bias=False)
#         (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       )
#     )
#     (1): BasicBlock(
#       (conv1): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), bias=False)
#       (bn2): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (shortcut): Sequential()
#     )
#   )
#   (layer3): Sequential(
#     (0): BasicBlock(
#       (conv1): Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
#       (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), bias=False)
#       (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (shortcut): Sequential(
#         (0): Conv2d(32, 64, kernel_size=(1, 1), stride=(2, 2), bias=False)
#         (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       )
#     )
#     (1): BasicBlock(
#       (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), bias=False)
#       (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (shortcut): Sequential()
#     )
#   )
#   (linear): Linear(in_features=64, out_features=10, bias=True)
# )

'''8. AE 모델 학습을 진행하며 학습 데이터에 대한 모델 성능을 확인하는 함수 정의'''
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
            print("Train Epoch : {} [{}/{}({:.0f}%)]\tTrain Loss : {:.6f}".format(
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
            output = model(image)
            test_loss += criterion(output, label)
            prediction = output.max(1, keepdim=True)[1]
            correct += prediction.eq(label.view_as(prediction)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, test_accuracy

'''10. AutoEncoder 학습을 실행시켜 Test set의 Reconstruction Error 확인하기'''
for Epoch in range(1, EPOCHS+1) :
    train(model, train_loader, optimizer, log_interval=200)
    test_loss, test_accuracy = evaluate(model, test_loader)
    print("\n[EPOCHS : {}], \tTest Loss : {:.4f}, \tTest Accuracy : {:.2f}%\n".format(
        Epoch, test_loss, test_accuracy
    ))

"""
Train Epoch : 1 [0/50000(0%)]   Train Loss : 2.370665
Train Epoch : 1 [6400/50000(13%)]       Train Loss : 1.795367
Train Epoch : 1 [12800/50000(26%)]      Train Loss : 1.240742
Train Epoch : 1 [19200/50000(38%)]      Train Loss : 1.284201
Train Epoch : 1 [25600/50000(51%)]      Train Loss : 1.024959
Train Epoch : 1 [32000/50000(64%)]      Train Loss : 1.641508
Train Epoch : 1 [38400/50000(77%)]      Train Loss : 1.105291
Train Epoch : 1 [44800/50000(90%)]      Train Loss : 0.984516

[EPOCHS : 1],   Test Loss : 0.0348,     Test Accuracy : 60.84%

Train Epoch : 2 [0/50000(0%)]   Train Loss : 1.041912
Train Epoch : 2 [6400/50000(13%)]       Train Loss : 0.854074
Train Epoch : 2 [12800/50000(26%)]      Train Loss : 1.125094
Train Epoch : 2 [19200/50000(38%)]      Train Loss : 0.861525
Train Epoch : 2 [25600/50000(51%)]      Train Loss : 0.948152
Train Epoch : 2 [32000/50000(64%)]      Train Loss : 0.962275
Train Epoch : 2 [38400/50000(77%)]      Train Loss : 0.728071
Train Epoch : 2 [44800/50000(90%)]      Train Loss : 0.907931

[EPOCHS : 2],   Test Loss : 0.0300,     Test Accuracy : 66.36%

Train Epoch : 3 [0/50000(0%)]   Train Loss : 0.787209
Train Epoch : 3 [6400/50000(13%)]       Train Loss : 0.754658
Train Epoch : 3 [12800/50000(26%)]      Train Loss : 0.872462
Train Epoch : 3 [19200/50000(38%)]      Train Loss : 0.526978
Train Epoch : 3 [25600/50000(51%)]      Train Loss : 0.508418
Train Epoch : 3 [32000/50000(64%)]      Train Loss : 0.775189
Train Epoch : 3 [38400/50000(77%)]      Train Loss : 0.733024
Train Epoch : 3 [44800/50000(90%)]      Train Loss : 0.792684

[EPOCHS : 3],   Test Loss : 0.0249,     Test Accuracy : 72.60%

Train Epoch : 4 [0/50000(0%)]   Train Loss : 0.996619
Train Epoch : 4 [6400/50000(13%)]       Train Loss : 0.588760
Train Epoch : 4 [12800/50000(26%)]      Train Loss : 0.599972
Train Epoch : 4 [19200/50000(38%)]      Train Loss : 0.724400
Train Epoch : 4 [25600/50000(51%)]      Train Loss : 0.784244
Train Epoch : 4 [32000/50000(64%)]      Train Loss : 0.591189
Train Epoch : 4 [38400/50000(77%)]      Train Loss : 0.684201
Train Epoch : 4 [44800/50000(90%)]      Train Loss : 0.775942

[EPOCHS : 4],   Test Loss : 0.0194,     Test Accuracy : 78.27%

Train Epoch : 5 [0/50000(0%)]   Train Loss : 0.756646
Train Epoch : 5 [6400/50000(13%)]       Train Loss : 0.942735
Train Epoch : 5 [12800/50000(26%)]      Train Loss : 0.585460
Train Epoch : 5 [19200/50000(38%)]      Train Loss : 0.505186
Train Epoch : 5 [25600/50000(51%)]      Train Loss : 0.324058
Train Epoch : 5 [32000/50000(64%)]      Train Loss : 0.366143
Train Epoch : 5 [38400/50000(77%)]      Train Loss : 0.261469
Train Epoch : 5 [44800/50000(90%)]      Train Loss : 0.544626

[EPOCHS : 5],   Test Loss : 0.0186,     Test Accuracy : 79.70%

Train Epoch : 6 [0/50000(0%)]   Train Loss : 0.330591
Train Epoch : 6 [6400/50000(13%)]       Train Loss : 0.500794
Train Epoch : 6 [12800/50000(26%)]      Train Loss : 0.281337
Train Epoch : 6 [19200/50000(38%)]      Train Loss : 0.522546
Train Epoch : 6 [25600/50000(51%)]      Train Loss : 0.598780
Train Epoch : 6 [32000/50000(64%)]      Train Loss : 0.718194
Train Epoch : 6 [38400/50000(77%)]      Train Loss : 0.535782
Train Epoch : 6 [44800/50000(90%)]      Train Loss : 0.745707

[EPOCHS : 6],   Test Loss : 0.0180,     Test Accuracy : 79.81%

Train Epoch : 7 [0/50000(0%)]   Train Loss : 0.622858
Train Epoch : 7 [6400/50000(13%)]       Train Loss : 0.456621
Train Epoch : 7 [12800/50000(26%)]      Train Loss : 0.526416
Train Epoch : 7 [19200/50000(38%)]      Train Loss : 0.370489
Train Epoch : 7 [25600/50000(51%)]      Train Loss : 0.329387
Train Epoch : 7 [32000/50000(64%)]      Train Loss : 0.581442
Train Epoch : 7 [38400/50000(77%)]      Train Loss : 0.387818
Train Epoch : 7 [44800/50000(90%)]      Train Loss : 0.328619

[EPOCHS : 7],   Test Loss : 0.0167,     Test Accuracy : 81.73%

Train Epoch : 8 [0/50000(0%)]   Train Loss : 0.269781
Train Epoch : 8 [6400/50000(13%)]       Train Loss : 0.341837
Train Epoch : 8 [12800/50000(26%)]      Train Loss : 0.794654
Train Epoch : 8 [19200/50000(38%)]      Train Loss : 0.398171
Train Epoch : 8 [25600/50000(51%)]      Train Loss : 0.475969
Train Epoch : 8 [32000/50000(64%)]      Train Loss : 0.256525
Train Epoch : 8 [38400/50000(77%)]      Train Loss : 0.477875
Train Epoch : 8 [44800/50000(90%)]      Train Loss : 0.216140

[EPOCHS : 8],   Test Loss : 0.0181,     Test Accuracy : 80.21%

Train Epoch : 9 [0/50000(0%)]   Train Loss : 0.466293
Train Epoch : 9 [6400/50000(13%)]       Train Loss : 0.307950
Train Epoch : 9 [12800/50000(26%)]      Train Loss : 0.400641
Train Epoch : 9 [19200/50000(38%)]      Train Loss : 0.693549
Train Epoch : 9 [25600/50000(51%)]      Train Loss : 0.278355
Train Epoch : 9 [32000/50000(64%)]      Train Loss : 0.487462
Train Epoch : 9 [38400/50000(77%)]      Train Loss : 0.332039
Train Epoch : 9 [44800/50000(90%)]      Train Loss : 0.527941

[EPOCHS : 9],   Test Loss : 0.0158,     Test Accuracy : 83.18%

Train Epoch : 10 [0/50000(0%)]  Train Loss : 0.216970
Train Epoch : 10 [6400/50000(13%)]      Train Loss : 0.566319
Train Epoch : 10 [12800/50000(26%)]     Train Loss : 0.410664
Train Epoch : 10 [19200/50000(38%)]     Train Loss : 0.306477
Train Epoch : 10 [25600/50000(51%)]     Train Loss : 0.264761
Train Epoch : 10 [32000/50000(64%)]     Train Loss : 0.387114
Train Epoch : 10 [38400/50000(77%)]     Train Loss : 0.299329
Train Epoch : 10 [44800/50000(90%)]     Train Loss : 0.442433

[EPOCHS : 10],  Test Loss : 0.0166,     Test Accuracy : 81.98%
"""
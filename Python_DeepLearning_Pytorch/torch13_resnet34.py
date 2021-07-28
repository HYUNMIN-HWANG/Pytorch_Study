# ResNet34
# 파이토치에서 제공하고 있는 레퍼런스 모델 (ResNet34, AlexNet, VGG, SqueezeNet 등등)

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

'''6. 파이토치 내에서 제공하는 ResNet34 모델 불러온 후 FC 층 추가 및 Output 크기 설정하기'''
import torchvision.models as models
model = models.resnet34(pretrained=False)   # resnet34를 불러온다. / pretrained=False : 모델의 구조만 불러옴 / pretrained=True : 모델 구조&미리 학습된 파라미터 값을 함께 불러올 수 있음 
num_ftrs = model.fc.in_features             # Fully Connected Layer를 구성하고 있는 부분에 접근,  num_ftrs : FC layer 인풋에 해당하는 노드 수 를 저장한다.
model.fc = nn.Linear(num_ftrs, 10)          # 새로운 레이어 추가, 최종 아웃풋 10개 클래스
model = model.cuda()                        # DEVEICE 할당, model.to(DEVICE)


'''7. Optimizer, Objective Function 설정하기'''
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

print(model)
'''
ResNet(
  (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
  (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (relu): ReLU(inplace=True)
  (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
  (layer1): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    (1): BasicBlock(
      (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    (2): BasicBlock(
      (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (layer2): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (downsample): Sequential(
        (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): BasicBlock(
      (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    (2): BasicBlock(
      (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    (3): BasicBlock(
      (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (layer3): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (downsample): Sequential(
        (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): BasicBlock(
      (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    (2): BasicBlock(
      (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    (3): BasicBlock(
      (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    (4): BasicBlock(
      (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    (5): BasicBlock(
      (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (layer4): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (downsample): Sequential(
        (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): BasicBlock(
      (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    (2): BasicBlock(
      (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
  (fc): Linear(in_features=512, out_features=10, bias=True)
)
'''

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
# for Epoch in range(1, EPOCHS+1) :
#     train(model, train_loader, optimizer, log_interval=200)
#     test_loss, test_accuracy = evaluate(model, test_loader)
#     print("\n[EPOCHS : {}], \tTest Loss : {:.4f}, \tTest Accuracy : {:.2f}%\n".format(
#         Epoch, test_loss, test_accuracy
#     ))

"""
Train Epoch : 1 [0/50000(0%)]   Train Loss : 2.635853
Train Epoch : 1 [6400/50000(13%)]       Train Loss : 1.804927
Train Epoch : 1 [12800/50000(26%)]      Train Loss : 1.288328
Train Epoch : 1 [19200/50000(38%)]      Train Loss : 1.371491
Train Epoch : 1 [25600/50000(51%)]      Train Loss : 1.548291
Train Epoch : 1 [32000/50000(64%)]      Train Loss : 1.459435
Train Epoch : 1 [38400/50000(77%)]      Train Loss : 1.338531
Train Epoch : 1 [44800/50000(90%)]      Train Loss : 1.167042

[EPOCHS : 1],   Test Loss : 0.0385,     Test Accuracy : 57.06%

Train Epoch : 2 [0/50000(0%)]   Train Loss : 1.405378
Train Epoch : 2 [6400/50000(13%)]       Train Loss : 1.310795
Train Epoch : 2 [12800/50000(26%)]      Train Loss : 1.022920
Train Epoch : 2 [19200/50000(38%)]      Train Loss : 1.063481
Train Epoch : 2 [25600/50000(51%)]      Train Loss : 1.049339
Train Epoch : 2 [32000/50000(64%)]      Train Loss : 1.595020
Train Epoch : 2 [38400/50000(77%)]      Train Loss : 1.252267
Train Epoch : 2 [44800/50000(90%)]      Train Loss : 0.982576

[EPOCHS : 2],   Test Loss : 0.0332,     Test Accuracy : 63.31%

Train Epoch : 3 [0/50000(0%)]   Train Loss : 0.458115
Train Epoch : 3 [6400/50000(13%)]       Train Loss : 0.693959
Train Epoch : 3 [12800/50000(26%)]      Train Loss : 1.151735
Train Epoch : 3 [19200/50000(38%)]      Train Loss : 0.875833
Train Epoch : 3 [25600/50000(51%)]      Train Loss : 0.693015
Train Epoch : 3 [32000/50000(64%)]      Train Loss : 1.054512
Train Epoch : 3 [38400/50000(77%)]      Train Loss : 0.853045
Train Epoch : 3 [44800/50000(90%)]      Train Loss : 1.067672

[EPOCHS : 3],   Test Loss : 0.0303,     Test Accuracy : 67.09%

Train Epoch : 4 [0/50000(0%)]   Train Loss : 1.122655
Train Epoch : 4 [6400/50000(13%)]       Train Loss : 1.046467
Train Epoch : 4 [12800/50000(26%)]      Train Loss : 0.694389
Train Epoch : 4 [19200/50000(38%)]      Train Loss : 0.742972
Train Epoch : 4 [25600/50000(51%)]      Train Loss : 0.559936
Train Epoch : 4 [32000/50000(64%)]      Train Loss : 0.808349
Train Epoch : 4 [38400/50000(77%)]      Train Loss : 0.846791
Train Epoch : 4 [44800/50000(90%)]      Train Loss : 0.742803

[EPOCHS : 4],   Test Loss : 0.0262,     Test Accuracy : 70.88%

Train Epoch : 5 [0/50000(0%)]   Train Loss : 0.696200
Train Epoch : 5 [6400/50000(13%)]       Train Loss : 0.646004
Train Epoch : 5 [12800/50000(26%)]      Train Loss : 0.847776
Train Epoch : 5 [19200/50000(38%)]      Train Loss : 0.927531
Train Epoch : 5 [25600/50000(51%)]      Train Loss : 0.532538
Train Epoch : 5 [32000/50000(64%)]      Train Loss : 0.503518
Train Epoch : 5 [38400/50000(77%)]      Train Loss : 0.890153
Train Epoch : 5 [44800/50000(90%)]      Train Loss : 1.105396

[EPOCHS : 5],   Test Loss : 0.0227,     Test Accuracy : 74.77%

Train Epoch : 6 [0/50000(0%)]   Train Loss : 0.794245
Train Epoch : 6 [6400/50000(13%)]       Train Loss : 0.779291
Train Epoch : 6 [12800/50000(26%)]      Train Loss : 0.688492
Train Epoch : 6 [19200/50000(38%)]      Train Loss : 0.718980
Train Epoch : 6 [25600/50000(51%)]      Train Loss : 0.541667
Train Epoch : 6 [32000/50000(64%)]      Train Loss : 0.632104
Train Epoch : 6 [38400/50000(77%)]      Train Loss : 0.431402
Train Epoch : 6 [44800/50000(90%)]      Train Loss : 0.403833

[EPOCHS : 6],   Test Loss : 0.0224,     Test Accuracy : 75.69%

Train Epoch : 7 [0/50000(0%)]   Train Loss : 0.680052
Train Epoch : 7 [6400/50000(13%)]       Train Loss : 0.862181
Train Epoch : 7 [12800/50000(26%)]      Train Loss : 0.561380
Train Epoch : 7 [19200/50000(38%)]      Train Loss : 0.434595
Train Epoch : 7 [25600/50000(51%)]      Train Loss : 0.321160
Train Epoch : 7 [32000/50000(64%)]      Train Loss : 0.869983
Train Epoch : 7 [38400/50000(77%)]      Train Loss : 0.389605
Train Epoch : 7 [44800/50000(90%)]      Train Loss : 0.564787

[EPOCHS : 7],   Test Loss : 0.0218,     Test Accuracy : 76.28%

Train Epoch : 8 [0/50000(0%)]   Train Loss : 0.593176
Train Epoch : 8 [6400/50000(13%)]       Train Loss : 0.341729
Train Epoch : 8 [12800/50000(26%)]      Train Loss : 0.422638
Train Epoch : 8 [19200/50000(38%)]      Train Loss : 0.422633
Train Epoch : 8 [25600/50000(51%)]      Train Loss : 0.880975
Train Epoch : 8 [32000/50000(64%)]      Train Loss : 0.492330
Train Epoch : 8 [38400/50000(77%)]      Train Loss : 1.095389
Train Epoch : 8 [44800/50000(90%)]      Train Loss : 0.411591

[EPOCHS : 8],   Test Loss : 0.0595,     Test Accuracy : 65.04%

Train Epoch : 9 [0/50000(0%)]   Train Loss : 1.195244
Train Epoch : 9 [6400/50000(13%)]       Train Loss : 0.292352
Train Epoch : 9 [12800/50000(26%)]      Train Loss : 0.653507
Train Epoch : 9 [19200/50000(38%)]      Train Loss : 0.884458
Train Epoch : 9 [25600/50000(51%)]      Train Loss : 0.525884
Train Epoch : 9 [32000/50000(64%)]      Train Loss : 0.423474
Train Epoch : 9 [38400/50000(77%)]      Train Loss : 0.237242
Train Epoch : 9 [44800/50000(90%)]      Train Loss : 0.786590

[EPOCHS : 9],   Test Loss : 0.0204,     Test Accuracy : 78.05%

Train Epoch : 10 [0/50000(0%)]  Train Loss : 0.488709
Train Epoch : 10 [6400/50000(13%)]      Train Loss : 0.382895
Train Epoch : 10 [12800/50000(26%)]     Train Loss : 0.733860
Train Epoch : 10 [19200/50000(38%)]     Train Loss : 0.391470
Train Epoch : 10 [25600/50000(51%)]     Train Loss : 0.232567
Train Epoch : 10 [32000/50000(64%)]     Train Loss : 0.500831
Train Epoch : 10 [38400/50000(77%)]     Train Loss : 0.325708
Train Epoch : 10 [44800/50000(90%)]     Train Loss : 0.364596

[EPOCHS : 10],  Test Loss : 0.0275,     Test Accuracy : 70.48%
"""

'''11. ImageNet 데이터로 학습된 ResNet34 모델을 불러온 후 Fine Tuning 해보기'''
model = models.resnet34(pretrained=True)       # pretrained=True :  ImageNet 데이터를 잘 분류할 수 있도록 학습된 파라미터를 resnet34 모델에 적용한다.
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)
model = model.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for Epoch in range(1, EPOCHS+1):
    train(model, train_loader, optimizer, log_interval=200) 
    test_loss, test_accuracy = evaluate(model, test_loader)
    print("\n[EPOCHS : {}], \tTest Loss : {:.4f}, \tTest Accuracy : {:.2f}%\n".format(
        Epoch, test_loss, test_accuracy
    ))

"""
Train Epoch : 1 [0/50000(0%)]   Train Loss : 2.592642
Train Epoch : 1 [6400/50000(13%)]       Train Loss : 1.425829
Train Epoch : 1 [12800/50000(26%)]      Train Loss : 0.898706
Train Epoch : 1 [19200/50000(38%)]      Train Loss : 1.415850
Train Epoch : 1 [25600/50000(51%)]      Train Loss : 0.874834
Train Epoch : 1 [32000/50000(64%)]      Train Loss : 0.767687
Train Epoch : 1 [38400/50000(77%)]      Train Loss : 1.212483
Train Epoch : 1 [44800/50000(90%)]      Train Loss : 1.030772

[EPOCHS : 1],   Test Loss : 0.0265,     Test Accuracy : 71.05%

Train Epoch : 2 [0/50000(0%)]   Train Loss : 0.673388
Train Epoch : 2 [6400/50000(13%)]       Train Loss : 0.779376
Train Epoch : 2 [12800/50000(26%)]      Train Loss : 0.876733
Train Epoch : 2 [19200/50000(38%)]      Train Loss : 0.923378
Train Epoch : 2 [25600/50000(51%)]      Train Loss : 0.711073
Train Epoch : 2 [32000/50000(64%)]      Train Loss : 0.588964
Train Epoch : 2 [38400/50000(77%)]      Train Loss : 1.173794
Train Epoch : 2 [44800/50000(90%)]      Train Loss : 0.916624

[EPOCHS : 2],   Test Loss : 0.0243,     Test Accuracy : 73.62%

Train Epoch : 3 [0/50000(0%)]   Train Loss : 0.675723
Train Epoch : 3 [6400/50000(13%)]       Train Loss : 0.887585
Train Epoch : 3 [12800/50000(26%)]      Train Loss : 0.719317
Train Epoch : 3 [19200/50000(38%)]      Train Loss : 0.644100
Train Epoch : 3 [25600/50000(51%)]      Train Loss : 0.682058
Train Epoch : 3 [32000/50000(64%)]      Train Loss : 0.672989
Train Epoch : 3 [38400/50000(77%)]      Train Loss : 0.585855
Train Epoch : 3 [44800/50000(90%)]      Train Loss : 0.778838

[EPOCHS : 3],   Test Loss : 0.0217,     Test Accuracy : 76.42%

Train Epoch : 4 [0/50000(0%)]   Train Loss : 0.850807
Train Epoch : 4 [6400/50000(13%)]       Train Loss : 0.404218
Train Epoch : 4 [12800/50000(26%)]      Train Loss : 0.520076
Train Epoch : 4 [19200/50000(38%)]      Train Loss : 0.583168
Train Epoch : 4 [25600/50000(51%)]      Train Loss : 0.770546
Train Epoch : 4 [32000/50000(64%)]      Train Loss : 0.284305
Train Epoch : 4 [38400/50000(77%)]      Train Loss : 0.594804
Train Epoch : 4 [44800/50000(90%)]      Train Loss : 0.262694

[EPOCHS : 4],   Test Loss : 0.0236,     Test Accuracy : 75.46%

Train Epoch : 5 [0/50000(0%)]   Train Loss : 1.056572
Train Epoch : 5 [6400/50000(13%)]       Train Loss : 0.566323
Train Epoch : 5 [12800/50000(26%)]      Train Loss : 0.771768
Train Epoch : 5 [19200/50000(38%)]      Train Loss : 0.491224
Train Epoch : 5 [25600/50000(51%)]      Train Loss : 0.600011
Train Epoch : 5 [32000/50000(64%)]      Train Loss : 0.760957
Train Epoch : 5 [38400/50000(77%)]      Train Loss : 0.910906
Train Epoch : 5 [44800/50000(90%)]      Train Loss : 0.579377

[EPOCHS : 5],   Test Loss : 0.0178,     Test Accuracy : 80.78%

Train Epoch : 6 [0/50000(0%)]   Train Loss : 0.645534
Train Epoch : 6 [6400/50000(13%)]       Train Loss : 0.444894
Train Epoch : 6 [12800/50000(26%)]      Train Loss : 0.512431
Train Epoch : 6 [19200/50000(38%)]      Train Loss : 0.416837
Train Epoch : 6 [25600/50000(51%)]      Train Loss : 0.504081
Train Epoch : 6 [32000/50000(64%)]      Train Loss : 0.942346
Train Epoch : 6 [38400/50000(77%)]      Train Loss : 0.261237
Train Epoch : 6 [44800/50000(90%)]      Train Loss : 0.180330

[EPOCHS : 6],   Test Loss : 0.0297,     Test Accuracy : 79.67%

Train Epoch : 7 [0/50000(0%)]   Train Loss : 0.403753
Train Epoch : 7 [6400/50000(13%)]       Train Loss : 0.996891
Train Epoch : 7 [12800/50000(26%)]      Train Loss : 0.514586
Train Epoch : 7 [19200/50000(38%)]      Train Loss : 0.414241
Train Epoch : 7 [25600/50000(51%)]      Train Loss : 0.458583
Train Epoch : 7 [32000/50000(64%)]      Train Loss : 0.604791
Train Epoch : 7 [38400/50000(77%)]      Train Loss : 0.884147
Train Epoch : 7 [44800/50000(90%)]      Train Loss : 0.466511

[EPOCHS : 7],   Test Loss : 0.0210,     Test Accuracy : 80.37%

Train Epoch : 8 [0/50000(0%)]   Train Loss : 0.170051
Train Epoch : 8 [6400/50000(13%)]       Train Loss : 0.352742
Train Epoch : 8 [12800/50000(26%)]      Train Loss : 0.550723
Train Epoch : 8 [19200/50000(38%)]      Train Loss : 0.429252
Train Epoch : 8 [25600/50000(51%)]      Train Loss : 0.227340
Train Epoch : 8 [32000/50000(64%)]      Train Loss : 0.287542
Train Epoch : 8 [38400/50000(77%)]      Train Loss : 0.508334
Train Epoch : 8 [44800/50000(90%)]      Train Loss : 0.406435

[EPOCHS : 8],   Test Loss : 0.0180,     Test Accuracy : 81.42%

Train Epoch : 9 [0/50000(0%)]   Train Loss : 0.277286
Train Epoch : 9 [6400/50000(13%)]       Train Loss : 1.127585
Train Epoch : 9 [12800/50000(26%)]      Train Loss : 0.226208
Train Epoch : 9 [19200/50000(38%)]      Train Loss : 0.442712
Train Epoch : 9 [25600/50000(51%)]      Train Loss : 0.408519
Train Epoch : 9 [32000/50000(64%)]      Train Loss : 0.264875
Train Epoch : 9 [38400/50000(77%)]      Train Loss : 0.209084
Train Epoch : 9 [44800/50000(90%)]      Train Loss : 0.131079

[EPOCHS : 9],   Test Loss : 0.0182,     Test Accuracy : 81.46%

Train Epoch : 10 [0/50000(0%)]  Train Loss : 0.516325
Train Epoch : 10 [6400/50000(13%)]      Train Loss : 0.154395
Train Epoch : 10 [12800/50000(26%)]     Train Loss : 0.224620
Train Epoch : 10 [19200/50000(38%)]     Train Loss : 0.411055
Train Epoch : 10 [25600/50000(51%)]     Train Loss : 0.530854
Train Epoch : 10 [32000/50000(64%)]     Train Loss : 0.104559
Train Epoch : 10 [38400/50000(77%)]     Train Loss : 0.755762
Train Epoch : 10 [44800/50000(90%)]     Train Loss : 0.248330

[EPOCHS : 10],  Test Loss : 0.0185,     Test Accuracy : 81.89%
"""

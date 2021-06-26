# CIFAR10 , CNN 설계하기

'''1. Module Import'''
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import optim
from torch._C import TracingState
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
train_dataset = datasets.CIFAR10(
    root="../data/CIFAR_10",
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

test_dataset=datasets.CIFAR10(
    root="../data/CIFAR_10",
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

'''4. 데이터 확인하기 (1)'''
for (X_train, y_train) in train_loader :
    print("X_train : ", X_train.size(), "type : ", X_train.type())
    print("y_train : ", y_train.size(), "type : ", y_train.type())

# X_train :  torch.Size([16, 3, 32, 32]) type :  torch.FloatTensor
# y_train :  torch.Size([16]) type :  torch.LongTensor

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

'''6. Convolution Neural Network(CNN) 모델 설계하기'''
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3,  # 채널 수 (red, green, blue) = 3
            out_channels=8, # 해당 값만큼 depth가 정해진다. 
            kernel_size=3,  # filter의 크기를 설정해주는 부분, 3*3 filter 크기가 적용된다.
            padding=1       # 1로 설정 : 왼쪽/오른쪽/위/아래쪽 모두 1층으로 0으로 채워준다.
        )
        self.conv2 = nn.Conv2d(
            in_channels=8,  # conv1에서 filer의 개수를 8로 설정했기 때문에 동일하게 맞춰줌
            out_channels=16,
            kernel_size=3,
            padding=1
        )
        self.pool = nn.MaxPool2d(   # 2차원의 Feature Map 내에서 지정한 크기 내 가장 큰 Feature Map 값만 사용하겠다.
            kernel_size=2,          # 2*2 filter가 돌아다니면서 가장 큰 Feature Map을 추출한다.
            stride=2                # 2단위로 움직임
        )
        self.fc1 = nn.Linear(8*8*16, 64)    # 1차원으로 펼친 후 여러 층의 FC layer를 통과시켜 분류합니다.
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 10)

    def forward(self, x) :
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)

        x = x.view(-1, 8*8*16)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x =self.fc3(x)
        x = F.log_softmax(x)
        return x


'''7. Optimizer, Objective Function 설정하기'''
model = CNN().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

print(model)
# CNN(
#   (conv1): Conv2d(3, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (conv2): Conv2d(8, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#   (fc1): Linear(in_features=1024, out_features=64, bias=True)
#   (fc2): Linear(in_features=64, out_features=32, bias=True)
#   (fc3): Linear(in_features=32, out_features=10, bias=True)
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
Train Epoch : 1 [0/50000(0%)]   Train Loss : 2.305091
Train Epoch : 1 [6400/50000(13%)]       Train Loss : 2.064333
Train Epoch : 1 [12800/50000(26%)]      Train Loss : 1.862936
Train Epoch : 1 [19200/50000(38%)]      Train Loss : 1.347729
Train Epoch : 1 [25600/50000(51%)]      Train Loss : 1.497383
Train Epoch : 1 [32000/50000(64%)]      Train Loss : 1.737059
Train Epoch : 1 [38400/50000(77%)]      Train Loss : 1.564931
Train Epoch : 1 [44800/50000(90%)]      Train Loss : 1.750263

[EPOCHS : 1],   Test Loss : 0.0473,     Test Accuracy : 45.74%

Train Epoch : 2 [0/50000(0%)]   Train Loss : 1.458806
Train Epoch : 2 [6400/50000(13%)]       Train Loss : 1.566024
Train Epoch : 2 [12800/50000(26%)]      Train Loss : 1.477793
Train Epoch : 2 [19200/50000(38%)]      Train Loss : 1.396083
Train Epoch : 2 [25600/50000(51%)]      Train Loss : 1.291113
Train Epoch : 2 [32000/50000(64%)]      Train Loss : 1.642197
Train Epoch : 2 [38400/50000(77%)]      Train Loss : 1.564679
Train Epoch : 2 [44800/50000(90%)]      Train Loss : 1.109997

[EPOCHS : 2],   Test Loss : 0.0407,     Test Accuracy : 53.00%

Train Epoch : 3 [0/50000(0%)]   Train Loss : 1.736112
Train Epoch : 3 [6400/50000(13%)]       Train Loss : 1.250734
Train Epoch : 3 [12800/50000(26%)]      Train Loss : 1.192211
Train Epoch : 3 [19200/50000(38%)]      Train Loss : 1.055193
Train Epoch : 3 [25600/50000(51%)]      Train Loss : 1.493732
Train Epoch : 3 [32000/50000(64%)]      Train Loss : 1.628249
Train Epoch : 3 [38400/50000(77%)]      Train Loss : 1.075507
Train Epoch : 3 [44800/50000(90%)]      Train Loss : 1.186393

[EPOCHS : 3],   Test Loss : 0.0385,     Test Accuracy : 56.40%

Train Epoch : 4 [0/50000(0%)]   Train Loss : 1.362184
Train Epoch : 4 [6400/50000(13%)]       Train Loss : 1.300152
Train Epoch : 4 [12800/50000(26%)]      Train Loss : 1.187189
Train Epoch : 4 [19200/50000(38%)]      Train Loss : 1.012689
Train Epoch : 4 [25600/50000(51%)]      Train Loss : 0.989973
Train Epoch : 4 [32000/50000(64%)]      Train Loss : 1.236058
Train Epoch : 4 [38400/50000(77%)]      Train Loss : 0.889755
Train Epoch : 4 [44800/50000(90%)]      Train Loss : 1.294611

[EPOCHS : 4],   Test Loss : 0.0362,     Test Accuracy : 58.95%

Train Epoch : 5 [0/50000(0%)]   Train Loss : 1.493649
Train Epoch : 5 [6400/50000(13%)]       Train Loss : 1.149936
Train Epoch : 5 [12800/50000(26%)]      Train Loss : 0.810156
Train Epoch : 5 [19200/50000(38%)]      Train Loss : 1.515345
Train Epoch : 5 [25600/50000(51%)]      Train Loss : 1.353247
Train Epoch : 5 [32000/50000(64%)]      Train Loss : 0.889964
Train Epoch : 5 [38400/50000(77%)]      Train Loss : 1.071889
Train Epoch : 5 [44800/50000(90%)]      Train Loss : 1.213433

[EPOCHS : 5],   Test Loss : 0.0348,     Test Accuracy : 60.75%

Train Epoch : 6 [0/50000(0%)]   Train Loss : 1.025747
Train Epoch : 6 [6400/50000(13%)]       Train Loss : 0.741907
Train Epoch : 6 [12800/50000(26%)]      Train Loss : 1.345846
Train Epoch : 6 [19200/50000(38%)]      Train Loss : 1.191878
Train Epoch : 6 [25600/50000(51%)]      Train Loss : 1.397616
Train Epoch : 6 [32000/50000(64%)]      Train Loss : 1.133369
Train Epoch : 6 [38400/50000(77%)]      Train Loss : 1.400675
Train Epoch : 6 [44800/50000(90%)]      Train Loss : 1.087888

[EPOCHS : 6],   Test Loss : 0.0341,     Test Accuracy : 61.53%

Train Epoch : 7 [0/50000(0%)]   Train Loss : 1.066894
Train Epoch : 7 [6400/50000(13%)]       Train Loss : 0.996661
Train Epoch : 7 [12800/50000(26%)]      Train Loss : 0.865586
Train Epoch : 7 [19200/50000(38%)]      Train Loss : 0.816590
Train Epoch : 7 [25600/50000(51%)]      Train Loss : 1.031510
Train Epoch : 7 [32000/50000(64%)]      Train Loss : 0.925066
Train Epoch : 7 [38400/50000(77%)]      Train Loss : 0.705674
Train Epoch : 7 [44800/50000(90%)]      Train Loss : 0.867462

[EPOCHS : 7],   Test Loss : 0.0332,     Test Accuracy : 63.06%

Train Epoch : 8 [0/50000(0%)]   Train Loss : 0.830322
Train Epoch : 8 [6400/50000(13%)]       Train Loss : 1.059814
Train Epoch : 8 [12800/50000(26%)]      Train Loss : 0.638058
Train Epoch : 8 [19200/50000(38%)]      Train Loss : 0.993242
Train Epoch : 8 [25600/50000(51%)]      Train Loss : 0.696558
Train Epoch : 8 [32000/50000(64%)]      Train Loss : 0.851323
Train Epoch : 8 [38400/50000(77%)]      Train Loss : 1.248648
Train Epoch : 8 [44800/50000(90%)]      Train Loss : 0.822921

[EPOCHS : 8],   Test Loss : 0.0336,     Test Accuracy : 62.19%

Train Epoch : 9 [0/50000(0%)]   Train Loss : 1.123265
Train Epoch : 9 [6400/50000(13%)]       Train Loss : 1.147301
Train Epoch : 9 [12800/50000(26%)]      Train Loss : 1.139442
Train Epoch : 9 [19200/50000(38%)]      Train Loss : 0.848066
Train Epoch : 9 [25600/50000(51%)]      Train Loss : 0.743272
Train Epoch : 9 [32000/50000(64%)]      Train Loss : 0.798626
Train Epoch : 9 [38400/50000(77%)]      Train Loss : 0.970531
Train Epoch : 9 [44800/50000(90%)]      Train Loss : 0.779829

[EPOCHS : 9],   Test Loss : 0.0324,     Test Accuracy : 63.35%

Train Epoch : 10 [0/50000(0%)]  Train Loss : 0.886073
Train Epoch : 10 [6400/50000(13%)]      Train Loss : 0.947690
Train Epoch : 10 [12800/50000(26%)]     Train Loss : 1.282630
Train Epoch : 10 [19200/50000(38%)]     Train Loss : 1.137680
Train Epoch : 10 [25600/50000(51%)]     Train Loss : 0.854252
Train Epoch : 10 [32000/50000(64%)]     Train Loss : 0.673913
Train Epoch : 10 [38400/50000(77%)]     Train Loss : 0.786400
Train Epoch : 10 [44800/50000(90%)]     Train Loss : 0.770124

[EPOCHS : 10],  Test Loss : 0.0329,     Test Accuracy : 63.55%
"""
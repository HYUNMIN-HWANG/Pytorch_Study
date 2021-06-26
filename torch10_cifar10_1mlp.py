# CIFAR10 , MLP 설계하기

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

'''6. Multi Layer Perceptron(MLP) 모델 설계하기'''
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(32*32*3, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
    
    def forward(self, x) :
        x = x.view(-1, 32*32*3)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x


'''7. Optimizer, Objective Function 설정하기'''
model = Net().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

print(model)
# Net(
#   (fc1): Linear(in_features=3072, out_features=512, bias=True)
#   (fc2): Linear(in_features=512, out_features=256, bias=True)
#   (fc3): Linear(in_features=256, out_features=10, bias=True)
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
Train Epoch : 1 [0/50000(0%)]   Train Loss : 2.297986
Train Epoch : 1 [6400/50000(13%)]       Train Loss : 1.969694
Train Epoch : 1 [12800/50000(26%)]      Train Loss : 1.952295
Train Epoch : 1 [19200/50000(38%)]      Train Loss : 1.948197
Train Epoch : 1 [25600/50000(51%)]      Train Loss : 1.756610
Train Epoch : 1 [32000/50000(64%)]      Train Loss : 1.575338
Train Epoch : 1 [38400/50000(77%)]      Train Loss : 1.903828
Train Epoch : 1 [44800/50000(90%)]      Train Loss : 1.843145

[EPOCHS : 1],   Test Loss : 0.0541,     Test Accuracy : 37.70%

Train Epoch : 2 [0/50000(0%)]   Train Loss : 1.851464
Train Epoch : 2 [6400/50000(13%)]       Train Loss : 1.554927
Train Epoch : 2 [12800/50000(26%)]      Train Loss : 1.972088
Train Epoch : 2 [19200/50000(38%)]      Train Loss : 1.693432
Train Epoch : 2 [25600/50000(51%)]      Train Loss : 1.626464
Train Epoch : 2 [32000/50000(64%)]      Train Loss : 1.940393
Train Epoch : 2 [38400/50000(77%)]      Train Loss : 1.460281
Train Epoch : 2 [44800/50000(90%)]      Train Loss : 2.092154

[EPOCHS : 2],   Test Loss : 0.0504,     Test Accuracy : 42.88%

Train Epoch : 3 [0/50000(0%)]   Train Loss : 1.870571
Train Epoch : 3 [6400/50000(13%)]       Train Loss : 1.534657
Train Epoch : 3 [12800/50000(26%)]      Train Loss : 1.671419
Train Epoch : 3 [19200/50000(38%)]      Train Loss : 1.453732
Train Epoch : 3 [25600/50000(51%)]      Train Loss : 1.516360
Train Epoch : 3 [32000/50000(64%)]      Train Loss : 1.623206
Train Epoch : 3 [38400/50000(77%)]      Train Loss : 1.428604
Train Epoch : 3 [44800/50000(90%)]      Train Loss : 1.801981

[EPOCHS : 3],   Test Loss : 0.0486,     Test Accuracy : 44.54%

Train Epoch : 4 [0/50000(0%)]   Train Loss : 1.603926
Train Epoch : 4 [6400/50000(13%)]       Train Loss : 1.725374
Train Epoch : 4 [12800/50000(26%)]      Train Loss : 1.506530
Train Epoch : 4 [19200/50000(38%)]      Train Loss : 1.660489
Train Epoch : 4 [25600/50000(51%)]      Train Loss : 1.498464
Train Epoch : 4 [32000/50000(64%)]      Train Loss : 1.140503
Train Epoch : 4 [38400/50000(77%)]      Train Loss : 1.488400
Train Epoch : 4 [44800/50000(90%)]      Train Loss : 1.478757

[EPOCHS : 4],   Test Loss : 0.0482,     Test Accuracy : 44.96%

Train Epoch : 5 [0/50000(0%)]   Train Loss : 1.476456
Train Epoch : 5 [6400/50000(13%)]       Train Loss : 1.612412
Train Epoch : 5 [12800/50000(26%)]      Train Loss : 1.498882
Train Epoch : 5 [19200/50000(38%)]      Train Loss : 1.641123
Train Epoch : 5 [25600/50000(51%)]      Train Loss : 1.297650
Train Epoch : 5 [32000/50000(64%)]      Train Loss : 1.351407
Train Epoch : 5 [38400/50000(77%)]      Train Loss : 1.374444
Train Epoch : 5 [44800/50000(90%)]      Train Loss : 1.680966

[EPOCHS : 5],   Test Loss : 0.0474,     Test Accuracy : 46.07%

Train Epoch : 6 [0/50000(0%)]   Train Loss : 1.468114
Train Epoch : 6 [6400/50000(13%)]       Train Loss : 1.205203
Train Epoch : 6 [12800/50000(26%)]      Train Loss : 1.378463
Train Epoch : 6 [19200/50000(38%)]      Train Loss : 1.531590
Train Epoch : 6 [25600/50000(51%)]      Train Loss : 1.987282
Train Epoch : 6 [32000/50000(64%)]      Train Loss : 1.219270
Train Epoch : 6 [38400/50000(77%)]      Train Loss : 1.120442
Train Epoch : 6 [44800/50000(90%)]      Train Loss : 1.314709

[EPOCHS : 6],   Test Loss : 0.0468,     Test Accuracy : 47.04%

Train Epoch : 7 [0/50000(0%)]   Train Loss : 1.459384
Train Epoch : 7 [6400/50000(13%)]       Train Loss : 1.372937
Train Epoch : 7 [12800/50000(26%)]      Train Loss : 1.533975
Train Epoch : 7 [19200/50000(38%)]      Train Loss : 1.231956
Train Epoch : 7 [25600/50000(51%)]      Train Loss : 1.508086
Train Epoch : 7 [32000/50000(64%)]      Train Loss : 1.150671
Train Epoch : 7 [38400/50000(77%)]      Train Loss : 1.585732
Train Epoch : 7 [44800/50000(90%)]      Train Loss : 1.376252

[EPOCHS : 7],   Test Loss : 0.0454,     Test Accuracy : 48.19%

Train Epoch : 8 [0/50000(0%)]   Train Loss : 1.447328
Train Epoch : 8 [6400/50000(13%)]       Train Loss : 1.497431
Train Epoch : 8 [12800/50000(26%)]      Train Loss : 1.529855
Train Epoch : 8 [19200/50000(38%)]      Train Loss : 1.845534
Train Epoch : 8 [25600/50000(51%)]      Train Loss : 1.827464
Train Epoch : 8 [32000/50000(64%)]      Train Loss : 1.484578
Train Epoch : 8 [38400/50000(77%)]      Train Loss : 1.098862
Train Epoch : 8 [44800/50000(90%)]      Train Loss : 1.235667

[EPOCHS : 8],   Test Loss : 0.0481,     Test Accuracy : 45.73%

Train Epoch : 9 [0/50000(0%)]   Train Loss : 1.543514
Train Epoch : 9 [6400/50000(13%)]       Train Loss : 1.256188
Train Epoch : 9 [12800/50000(26%)]      Train Loss : 1.464668
Train Epoch : 9 [19200/50000(38%)]      Train Loss : 1.721856
Train Epoch : 9 [25600/50000(51%)]      Train Loss : 1.504124
Train Epoch : 9 [32000/50000(64%)]      Train Loss : 1.423015
Train Epoch : 9 [38400/50000(77%)]      Train Loss : 1.523070
Train Epoch : 9 [44800/50000(90%)]      Train Loss : 1.478538

[EPOCHS : 9],   Test Loss : 0.0456,     Test Accuracy : 48.49%

Train Epoch : 10 [0/50000(0%)]  Train Loss : 1.585778
Train Epoch : 10 [6400/50000(13%)]      Train Loss : 1.472077
Train Epoch : 10 [12800/50000(26%)]     Train Loss : 1.272221
Train Epoch : 10 [19200/50000(38%)]     Train Loss : 1.301446
Train Epoch : 10 [25600/50000(51%)]     Train Loss : 1.231540
Train Epoch : 10 [32000/50000(64%)]     Train Loss : 1.557837
Train Epoch : 10 [38400/50000(77%)]     Train Loss : 1.069491
Train Epoch : 10 [44800/50000(90%)]     Train Loss : 1.697077

[EPOCHS : 10],  Test Loss : 0.0454,     Test Accuracy : 48.07%
"""

'''1. Module Import'''
import numpy as np                              # 선형 대수와 관련된 함수를 ㅜ십게 이용할 수 있는 모듈
import matplotlib.pyplot as plt                 # 함수 실행 결과 산출물에 대한 수리츷 사람이 쉽게 이해할 수 있도록 시각화
import torch
import torch.nn as nn                           # 인공 신경망 모델을 설계할 때 필요한 함수를 모아 놓은 모듈
import torch.nn.functional as F
from torchvision import transforms, datasets

'''2. 딥러닝 모델을 설계할 떄 활용하는 장비 확인'''
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')   # GPU
else :
    DEVICE = torch.device('cpu')

print('Using PyTorch version : ', torch.__version__, ', Device : ', DEVICE)
# Using PyTorch version :  1.9.0 , Device :  cuda

BATCH_SIZE = 32
# MLP 모딜을 학습할 때 필요한 데이터 개수의 단위
# Mini-Batch 1개 단위에 데이터가 32개 구성되어 있는 것
# Mini-Batch 1개로 학습 1회를 진행함
# Iteration : Mini-Batch 1개를 이용해 학습하는 횟수, 전체 데이터 개수에서 Mini-Batch 1개를 구성하는 데이터 개수로 나눈 몫만큼
EPOCHS = 10
# Epoch : 전체 데이터를 이용해 학습을 진행한 횟수, Mini-Batch를 전부 이용하는 횟수

'''3. MNIST 데이터 다운로드(train, test set 분리하기)'''
train_dataset = datasets.MNIST(root='../data/MNIST',                # 데이터가 저장될 장소 지정
                                train=True,
                                download=True,
                                transform=transforms.ToTensor())    # ToTensor : 데이터를 tensor 형태로 변경, 0과 1사이 범위로 정규화
                            
test_dataset = datasets.MNIST(root='../data/MNIST',
                                train=False,
                                transform = transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                            batch_size=BATCH_SIZE,  # 데이터를 Mini-Batch 단위로 분리해 지정
                                            shuffle=True)           # 데이터를 섞어준다.

test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                            batch_size=BATCH_SIZE,
                                            shuffle=False)

'''4. 데이터 확인하기(1)'''
# Mini-Batch 단위로 할당한 데이터의 개수와 형태를 확인
for (X_train, y_train) in train_loader:
    print("X_train : ", X_train.size(), "type : ", X_train.type())
    print("y_train : ", y_train.size(), "type : ", y_train.type())

# X_train :  torch.Size([32, 1, 28, 28]) type :  torch.FloatTensor   << [mini-batch, channel, 가로, 세로]
# y_train :  torch.Size([32]) type :  torch.LongTensor


'''5. 데이터 확인하기(2)'''
pltsize = 1
plt.figure(figsize=(10*pltsize, pltsize))   # (10,1)크기의 그림판을 만든다.
for i in range(10):
    plt.subplot(1, 10, i+1)                 # i+1 번째에 그림을 넣는다.
    plt.axis('off')
    plt.imshow(X_train[i, :, :, :].numpy().reshape(28,28), cmap="gray_r")
    plt.title("Class : " + str(y_train[i].item()))
plt.show()


'''6. MLP(Multi Layer Perceptron) 모델 설계하기'''
class Net(nn.Module) :   # nn.Module 클래스를 상속받은, Net 클래스 정의
    def __init__(self):
        super(Net, self).__init__()         # nn.Module 내에 있는 메서드를 상속받아 이용함
        self.fc1 = nn.Linear(28*28, 512)    # 첫번째 Fully Connected Layer - input(28*28*1), 다음FC의 노드수(512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)       # 총 10가지 클래스를 표현하기 위한 Label 값

    def forward(self, x):
        x = x.view(-1, 28*28)       # 1차원의 벡터 값을 입력으로 받을 수 있다. 28*28 2차원 데이터를 view를 통해서 1차원으로 만들어준다. => flatten
        x = self.fc1(x)             # 첫번째 Fully Connected Layer
        x = F.sigmoid(x)            # 비활성화 함수 sigmoid
        x = self.fc2(x)             # 두번째 Fully Connected Layer
        x = F.sigmoid(x)            # 비활성화 함수 sigmoid
        x = self.fc3(x)             # 세번째 Fully Connected Layer
        x = F.log_softmax(x, dim=1) # 총 10가지 경우의 수 중 하나로 분류하는 일을 수행, softmax를 통해서 확률 값을 계산한다.& log를 사용함으로써 loss 값에 대한 gradient 값을 좀 더 원활하게 계산할 수 있다.
        return x                    # 최종 output

'''7. Optimizer, Objective Funtion'''
model = Net().to(DEVICE)            # net 모델을 DEVICE에 할당한다.
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)  # 파라미터 업데이트할 때 이용하는 optimizer 설정
criterion = nn.CrossEntropyLoss()   # Class를 원핫인코딩 값

print(model)
# Net(
#   (fc1): Linear(in_features=784, out_features=512, bias=True)
#   (fc2): Linear(in_features=512, out_features=256, bias=True)
#   (fc3): Linear(in_features=256, out_features=10, bias=True)
# )


'''8. MLP 모델 학습을 진행하며 학습 데이터에 대한 모델 성능을 확인하는 함수 정의'''
def train(model, train_loader, optimizer, log_interval) :
    model.train()   # 기존에 정의한 MLP 모델을 학습 상태로 지정함
    for batch_idx, (image, label) in enumerate (train_loader):
        image = image.to(DEVICE)
        label = label.to(DEVICE)
        optimizer.zero_grad()           # gradient 초기화
        output = model(image)           # image를 input으로 이용해 output을 계산함
        loss = criterion(output, label) # CrossEntropyLoss를 이용해 loss를 구한다.
        loss.backward()                 # bach propagation을 통해 계산된 gradient 값을 각 파라미터에 할당
        optimizer.step()                # 각 파리미터에 할당된 gradient 값을 이용해 파라미터 값을 업데이터 한다.

        if batch_idx % log_interval == 0 :
            print("Train Epoch : {} [{}/{}({:.0f}%)]\t Train loss : {:.6f}".format(
                Epoch, batch_idx * len(image), len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.item() 
            ))

'''9. 학습되는 과정 속에서 검증 데이터에 대한 모델 성능을 확인하는 함수 정의'''
def evaluate(model, test_loader) :
    model.eval()        # 훈련상태가 아닌 평가 상태로 지정함
    test_loss = 0
    correct = 0

    with torch.no_grad() :  # gradient 가 업데이터되는 현상을 방지하기 위함
        for image, label in test_loader :
            image = image.to(DEVICE)
            label = label.to(DEVICE)
            output = model(image)
            test_loss += criterion(output, label).item()    # loss 계산
            prediction = output.max(1, keepdim=True)[1]     # output 10인 벡터 값이 나온다. 계산된 벡터 값 내 가장 큰 값인 위치에 대해 해당 위치에 대응하는 클래스로 예측했다고 판단한다.
            correct += prediction.eq(label.view_as(prediction)).sum().item() # 예측한 클래스와 실제 클래스가 같으면 correct 횟수로 저장한다.
        
    test_loss /= len(test_loader.dataset)   # test loss 값을 test loader 내에 존재하는 mini batch 개수만큼 나눠 loss의 평균을 구한다.
    test_accuracy = 100. * correct / len(test_loader.dataset)   # test_loader 중에서 몇 개 맞췄는지 accuracy를 계산한다.
    return test_loss, test_accuracy
    

'''10. MLP 학습을 실행하면서 train, test set의 loss 및 test set accuracy 확인하기'''
for Epoch in range(1, EPOCHS + 1) :
    train(model, train_loader, optimizer, log_interval = 200)   # 정의한 train 함수를 실행한다 / log_interval : 학습이 진행되면서 mini batch의 index를 이용해 과정을 모니터링할 수 있도록 출력하는 것을 의미함
    test_loss, test_accuracy = evaluate(model, test_loader)
    print("\n[EPOCH : {}], \tTEST LOSS : {:.4f}, \tTEST ACCURACY : {:.2f} %\n".format(
        Epoch, test_loss, test_accuracy))



print("End")

'''
Train Epoch : 1 [0/60000(0%)]    Train loss : 2.339285
Train Epoch : 1 [6400/60000(11%)]        Train loss : 2.312460
Train Epoch : 1 [12800/60000(21%)]       Train loss : 2.295702
Train Epoch : 1 [19200/60000(32%)]       Train loss : 2.268482
Train Epoch : 1 [25600/60000(43%)]       Train loss : 2.287223
Train Epoch : 1 [32000/60000(53%)]       Train loss : 2.306753
Train Epoch : 1 [38400/60000(64%)]       Train loss : 2.319535
Train Epoch : 1 [44800/60000(75%)]       Train loss : 2.240298
Train Epoch : 1 [51200/60000(85%)]       Train loss : 2.274410
Train Epoch : 1 [57600/60000(96%)]       Train loss : 2.337006

[EPOCH : 1],    TEST LOSS : 0.0698,     TEST ACCURACY : 29.07 %

Train Epoch : 2 [0/60000(0%)]    Train loss : 2.216220
Train Epoch : 2 [6400/60000(11%)]        Train loss : 2.209300
Train Epoch : 2 [12800/60000(21%)]       Train loss : 2.152533
Train Epoch : 2 [19200/60000(32%)]       Train loss : 2.108081
Train Epoch : 2 [25600/60000(43%)]       Train loss : 2.019098
Train Epoch : 2 [32000/60000(53%)]       Train loss : 1.888179
Train Epoch : 2 [38400/60000(64%)]       Train loss : 1.649369
Train Epoch : 2 [44800/60000(75%)]       Train loss : 1.613809
Train Epoch : 2 [51200/60000(85%)]       Train loss : 1.384196
Train Epoch : 2 [57600/60000(96%)]       Train loss : 1.254256

[EPOCH : 2],    TEST LOSS : 0.0390,     TEST ACCURACY : 63.48 %

Train Epoch : 3 [0/60000(0%)]    Train loss : 1.030868
Train Epoch : 3 [6400/60000(11%)]        Train loss : 1.202375
Train Epoch : 3 [12800/60000(21%)]       Train loss : 0.975588
Train Epoch : 3 [19200/60000(32%)]       Train loss : 1.005746
Train Epoch : 3 [25600/60000(43%)]       Train loss : 0.917890
Train Epoch : 3 [32000/60000(53%)]       Train loss : 0.887119
Train Epoch : 3 [38400/60000(64%)]       Train loss : 0.801069
Train Epoch : 3 [44800/60000(75%)]       Train loss : 0.930210
Train Epoch : 3 [51200/60000(85%)]       Train loss : 0.673280
Train Epoch : 3 [57600/60000(96%)]       Train loss : 0.873731

[EPOCH : 3],    TEST LOSS : 0.0236,     TEST ACCURACY : 77.47 %

Train Epoch : 4 [0/60000(0%)]    Train loss : 0.471017
Train Epoch : 4 [6400/60000(11%)]        Train loss : 0.730787
Train Epoch : 4 [12800/60000(21%)]       Train loss : 0.781091
Train Epoch : 4 [19200/60000(32%)]       Train loss : 0.841945
Train Epoch : 4 [25600/60000(43%)]       Train loss : 0.498126
Train Epoch : 4 [32000/60000(53%)]       Train loss : 0.650484
Train Epoch : 4 [38400/60000(64%)]       Train loss : 0.644907
Train Epoch : 4 [44800/60000(75%)]       Train loss : 0.538706
Train Epoch : 4 [51200/60000(85%)]       Train loss : 0.921294
Train Epoch : 4 [57600/60000(96%)]       Train loss : 0.404870

[EPOCH : 4],    TEST LOSS : 0.0174,     TEST ACCURACY : 83.68 %

Train Epoch : 5 [0/60000(0%)]    Train loss : 0.495194
Train Epoch : 5 [6400/60000(11%)]        Train loss : 0.448802
Train Epoch : 5 [12800/60000(21%)]       Train loss : 0.402423
Train Epoch : 5 [19200/60000(32%)]       Train loss : 0.390460
Train Epoch : 5 [25600/60000(43%)]       Train loss : 0.358794
Train Epoch : 5 [32000/60000(53%)]       Train loss : 0.560431
Train Epoch : 5 [38400/60000(64%)]       Train loss : 0.403852
Train Epoch : 5 [44800/60000(75%)]       Train loss : 0.754409
Train Epoch : 5 [51200/60000(85%)]       Train loss : 0.526144
Train Epoch : 5 [57600/60000(96%)]       Train loss : 0.834692

[EPOCH : 5],    TEST LOSS : 0.0143,     TEST ACCURACY : 86.85 %

Train Epoch : 6 [0/60000(0%)]    Train loss : 0.532759
Train Epoch : 6 [6400/60000(11%)]        Train loss : 0.210242
Train Epoch : 6 [12800/60000(21%)]       Train loss : 0.586309
Train Epoch : 6 [19200/60000(32%)]       Train loss : 0.475480
Train Epoch : 6 [25600/60000(43%)]       Train loss : 0.258359
Train Epoch : 6 [32000/60000(53%)]       Train loss : 0.283448
Train Epoch : 6 [38400/60000(64%)]       Train loss : 0.549067
Train Epoch : 6 [44800/60000(75%)]       Train loss : 0.224034
Train Epoch : 6 [51200/60000(85%)]       Train loss : 0.497705
Train Epoch : 6 [57600/60000(96%)]       Train loss : 0.505577

[EPOCH : 6],    TEST LOSS : 0.0127,     TEST ACCURACY : 88.27 %

Train Epoch : 7 [0/60000(0%)]    Train loss : 0.345364
Train Epoch : 7 [6400/60000(11%)]        Train loss : 0.548752
Train Epoch : 7 [12800/60000(21%)]       Train loss : 0.450934
Train Epoch : 7 [19200/60000(32%)]       Train loss : 0.468907
Train Epoch : 7 [25600/60000(43%)]       Train loss : 0.533966
Train Epoch : 7 [32000/60000(53%)]       Train loss : 0.144551
Train Epoch : 7 [38400/60000(64%)]       Train loss : 0.389049
Train Epoch : 7 [44800/60000(75%)]       Train loss : 0.724590
Train Epoch : 7 [51200/60000(85%)]       Train loss : 0.579648
Train Epoch : 7 [57600/60000(96%)]       Train loss : 0.638883

[EPOCH : 7],    TEST LOSS : 0.0120,     TEST ACCURACY : 89.01 %

Train Epoch : 8 [0/60000(0%)]    Train loss : 0.394323
Train Epoch : 8 [6400/60000(11%)]        Train loss : 0.399794
Train Epoch : 8 [12800/60000(21%)]       Train loss : 0.421491
Train Epoch : 8 [19200/60000(32%)]       Train loss : 0.599039
Train Epoch : 8 [25600/60000(43%)]       Train loss : 0.281001
Train Epoch : 8 [32000/60000(53%)]       Train loss : 0.170169
Train Epoch : 8 [38400/60000(64%)]       Train loss : 0.188152
Train Epoch : 8 [44800/60000(75%)]       Train loss : 0.391346
Train Epoch : 8 [51200/60000(85%)]       Train loss : 0.448238
Train Epoch : 8 [57600/60000(96%)]       Train loss : 0.282045

[EPOCH : 8],    TEST LOSS : 0.0113,     TEST ACCURACY : 89.46 %

Train Epoch : 9 [0/60000(0%)]    Train loss : 0.505396
Train Epoch : 9 [6400/60000(11%)]        Train loss : 0.447174
Train Epoch : 9 [12800/60000(21%)]       Train loss : 0.710079
Train Epoch : 9 [19200/60000(32%)]       Train loss : 0.356599
Train Epoch : 9 [25600/60000(43%)]       Train loss : 0.284339
Train Epoch : 9 [32000/60000(53%)]       Train loss : 0.245319
Train Epoch : 9 [38400/60000(64%)]       Train loss : 0.503830
Train Epoch : 9 [44800/60000(75%)]       Train loss : 0.915378
Train Epoch : 9 [51200/60000(85%)]       Train loss : 0.308502
Train Epoch : 9 [57600/60000(96%)]       Train loss : 0.364901

[EPOCH : 9],    TEST LOSS : 0.0108,     TEST ACCURACY : 89.97 %

Train Epoch : 10 [0/60000(0%)]   Train loss : 0.600254
Train Epoch : 10 [6400/60000(11%)]       Train loss : 0.281672
Train Epoch : 10 [12800/60000(21%)]      Train loss : 0.293459
Train Epoch : 10 [19200/60000(32%)]      Train loss : 0.397549
Train Epoch : 10 [25600/60000(43%)]      Train loss : 0.297046
Train Epoch : 10 [32000/60000(53%)]      Train loss : 0.382541
Train Epoch : 10 [38400/60000(64%)]      Train loss : 0.083765
Train Epoch : 10 [44800/60000(75%)]      Train loss : 0.351984
Train Epoch : 10 [51200/60000(85%)]      Train loss : 0.474943
Train Epoch : 10 [57600/60000(96%)]      Train loss : 0.275839

[EPOCH : 10],   TEST LOSS : 0.0105,     TEST ACCURACY : 90.25 %
'''
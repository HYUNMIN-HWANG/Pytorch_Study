# 원본 데이터를 생성하는 Autoencoder 실습

'''1. Module Import'''
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets


'''2. 딥러닝 모델을 설계할 때 활용하는 장비 확인'''
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else :
    DEVICE = torch.device("cpu")

print(torch.__version__, DEVICE)    # 1.9.0 cuda

BATCH_SIZE = 32
# AE를 학습할 때 필요한 데이터 개수의 단위
EPOCHS = 10
# mini-batch 1개 단위로 back propagation을 이용해 AE 가중값을 업데이트 하는데, Epoch은 존재하고 있는 mini-batch를 전부 이용하는 횟수를 의미함

'''3. FashionMNIST 데이터 다운로드 (train, test set 분리하기)'''
train_dataset = datasets.FashionMNIST(
    root='../data/FashionMNIST',
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

test_dataset = datasets.FashionMNIST(
    root = '../data/FashionMNIST',
    train=False,
    transform=transforms.ToTensor()
)

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

'''4. 데이터 확인하기 (1)'''
# Mini Batch 단위로 할당한 데이터의 개수와 데이터 형태를 확인한다.
for (X_train, y_train) in train_loader :
    print("X_train : ", X_train.size(), "type : ", X_train.type())
    print("y_train : ", y_train.size(), "type : ", y_train.type())

# X_train :  torch.Size([32, 1, 28, 28]) type :  torch.FloatTensor
# y_train :  torch.Size([32]) type :  torch.LongTensor


'''5. 데이터 확인하기 (2)'''
# 데이터를 직접 확인할 수 있는 시각화 코드
pltsize=1
plt.figure(figsize=(10 * pltsize, pltsize))
for i in range(10) :
    plt.subplot(1, 10, i+1)
    plt.axis('off')
    plt.imshow(X_train[i,:,:,:].numpy().reshape(28,28), cmap="gray_r")
    plt.title("Class : " + str(y_train[i].item()))
plt.show()

'''6. AutoEncoder(AE) 모델 설계하기'''
class AE(nn.Module) :
    def __init__(self) :
        super(AE, self).__init__()
        self.encoder = nn.Sequential(       # 인코더 정의
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
        )
        self.decoder = nn.Sequential(       # 디코더 정의
            nn.Linear(32, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 28*28),
        )

    def forward(self, x) :
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

'''7. Optimizer, Objective Function 설정하기'''
model = AE().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)    # back propagation을 통해서 파라미터를 업데이트할 때 이용
criterion = nn.MSELoss()       # MSE (MeanSquaredError)

print(model)
'''
AE(
  (encoder): Sequential(
    (0): Linear(in_features=784, out_features=512, bias=True)
    (1): ReLU()
    (2): Linear(in_features=512, out_features=256, bias=True)
    (3): ReLU()
    (4): Linear(in_features=256, out_features=32, bias=True)
  )
  (decoder): Sequential(
    (0): Linear(in_features=32, out_features=256, bias=True)
    (1): ReLU()
    (2): Linear(in_features=256, out_features=512, bias=True)
    (3): ReLU()
    (4): Linear(in_features=512, out_features=784, bias=True)
  )
)
'''


'''8. AE 모델 학습을 진행하며 학습 데이터에 대한 모델 성능을 확인하는 함수 정의'''
def train(model, train_loader, optimizer, log_interval) :
    model.train()
    for batch_idx, (image, _) in enumerate(train_loader) :
        image = image.view(-1, 28*28).to(DEVICE)        # 2차원 이미지를 1차원 데이터로 재구성한 후 할당
        target = image.view(-1, 28*28).to(DEVICE)
        optimizer.zero_grad()                           # optimizer의 gradient를 초기화함
        encoded, decoded = model(image)                 # image를 input으로 하여 output 계산
        loss = criterion(decoded, target)               # MSE로 loss 계산
        loss.backward()                                 # Back Propagation
        optimizer.step()

        if batch_idx % log_interval == 0 :
            print("Train Epoch : {} [{}/{} ({:.0f}%)]\tTrain Loss : {:.6f}".format(
                Epoch, batch_idx*len(image), len(train_loader.dataset), 100*batch_idx/len(train_loader), loss.item()
            ))

'''9. 학습되는 과정 속에서 검증 데이터에 대한 모델 성능을 확인하는 함수 정의'''
def evaluate(model, test_loader) :
    model.eval()        # 성능 평가 모델
    test_loss = 0
    real_image = []     # 실제 이미지 데이터를 리스트에 저장하기 위함
    gen_image = []      # 생성되는 이미지 데이터를 리스트에 저장하기 위함

    with torch.no_grad() :  # gradient를 통해서 파라미터 값이 업데이트 되는 것을 방지
        for image, _ in test_loader :
            image = image.view(-1, 28*28).to(DEVICE)
            target = image.view(-1, 28*28).to(DEVICE)
            encoded, decoded = model(image)

            test_loss += criterion(decoded, image).item()   # MSE loss 값 계산해서 loss 값 갱신
            real_image.append(image.to("cpu"))              # 실제 이미지를 리스트에 추가
            gen_image.append(decoded.to("cpu"))             # AE모델을 통해서 생성된 이미지를 리스트에 추가

    test_loss /= len(test_loader.dataset)
    return test_loss, real_image, gen_image

    
'''10. AutoEncoder 학습을 실행시켜 Test set의 Reconstruction Error 확인하기'''
for Epoch in range(1, EPOCHS+1):                                    # EPOCH 수만큼 학습을 진행
    train(model, train_loader, optimizer, log_interval=200)         # train 함수 실행
    test_loss, real_image, gen_image = evaluate(model, test_loader) 
    print("\n[EPOCH : {}], \tTEST LOSS : {:.4f}".format(
        Epoch, test_loss
    ))

    f, a = plt.subplots(2, 10, figsize=(10, 4))                     # 실제 이미지와 생성된 이미지를 비교해 학습의 진행도를 확인할 수 있음
    for i in range(10):
        img = np.reshape(real_image[0][i], (28,28))
        a[0][i].imshow(img, cmap="gray_r")
        a[0][i].set_xticks(())  # x축의 눈금을 내가 원하는 범위, 간격으로 지정할 수 있다.
        a[0][i].set_yticks(())  # y축의 눈금을 내가 원하는 범위, 간격으로 지정할 수 있다.

    for i in range(10):
        img = np.reshape(gen_image[0][i], (28,28))
        a[1][i].imshow(img, cmap="gray_r")
        a[1][i].set_xticks(())
        a[1][i].set_yticks(())

    plt.show()

'''
Train Epoch : 1 [0/60000 (0%)]  Train Loss : 0.217224
Train Epoch : 1 [6400/60000 (11%)]      Train Loss : 0.023795
Train Epoch : 1 [12800/60000 (21%)]     Train Loss : 0.023814
Train Epoch : 1 [19200/60000 (32%)]     Train Loss : 0.020533
Train Epoch : 1 [25600/60000 (43%)]     Train Loss : 0.021273
Train Epoch : 1 [32000/60000 (53%)]     Train Loss : 0.016986
Train Epoch : 1 [38400/60000 (64%)]     Train Loss : 0.018766
Train Epoch : 1 [44800/60000 (75%)]     Train Loss : 0.014232
Train Epoch : 1 [51200/60000 (85%)]     Train Loss : 0.012027
Train Epoch : 1 [57600/60000 (96%)]     Train Loss : 0.013546

[EPOCH : 1],    TEST LOSS : 0.0005
Train Epoch : 2 [0/60000 (0%)]  Train Loss : 0.017922
Train Epoch : 2 [6400/60000 (11%)]      Train Loss : 0.017862
Train Epoch : 2 [12800/60000 (21%)]     Train Loss : 0.018071
Train Epoch : 2 [19200/60000 (32%)]     Train Loss : 0.012583
Train Epoch : 2 [25600/60000 (43%)]     Train Loss : 0.016994
Train Epoch : 2 [32000/60000 (53%)]     Train Loss : 0.012052
Train Epoch : 2 [38400/60000 (64%)]     Train Loss : 0.011872
Train Epoch : 2 [44800/60000 (75%)]     Train Loss : 0.014327
Train Epoch : 2 [51200/60000 (85%)]     Train Loss : 0.014027
Train Epoch : 2 [57600/60000 (96%)]     Train Loss : 0.013846

[EPOCH : 2],    TEST LOSS : 0.0004
Train Epoch : 3 [0/60000 (0%)]  Train Loss : 0.010826
Train Epoch : 3 [6400/60000 (11%)]      Train Loss : 0.015684
Train Epoch : 3 [12800/60000 (21%)]     Train Loss : 0.010941
Train Epoch : 3 [19200/60000 (32%)]     Train Loss : 0.010692
Train Epoch : 3 [25600/60000 (43%)]     Train Loss : 0.015735
Train Epoch : 3 [32000/60000 (53%)]     Train Loss : 0.011278
Train Epoch : 3 [38400/60000 (64%)]     Train Loss : 0.013062
Train Epoch : 3 [44800/60000 (75%)]     Train Loss : 0.010077
Train Epoch : 3 [51200/60000 (85%)]     Train Loss : 0.011284
Train Epoch : 3 [57600/60000 (96%)]     Train Loss : 0.013678

[EPOCH : 3],    TEST LOSS : 0.0004
Train Epoch : 4 [0/60000 (0%)]  Train Loss : 0.011689
Train Epoch : 4 [6400/60000 (11%)]      Train Loss : 0.012160
Train Epoch : 4 [12800/60000 (21%)]     Train Loss : 0.012927
Train Epoch : 4 [19200/60000 (32%)]     Train Loss : 0.010529
Train Epoch : 4 [25600/60000 (43%)]     Train Loss : 0.014126
Train Epoch : 4 [32000/60000 (53%)]     Train Loss : 0.010170
Train Epoch : 4 [38400/60000 (64%)]     Train Loss : 0.010419
Train Epoch : 4 [44800/60000 (75%)]     Train Loss : 0.010642
Train Epoch : 4 [51200/60000 (85%)]     Train Loss : 0.010550
Train Epoch : 4 [57600/60000 (96%)]     Train Loss : 0.012011

[EPOCH : 4],    TEST LOSS : 0.0004
Train Epoch : 5 [0/60000 (0%)]  Train Loss : 0.012466
Train Epoch : 5 [6400/60000 (11%)]      Train Loss : 0.009412
Train Epoch : 5 [12800/60000 (21%)]     Train Loss : 0.011348
Train Epoch : 5 [19200/60000 (32%)]     Train Loss : 0.011488
Train Epoch : 5 [25600/60000 (43%)]     Train Loss : 0.011727
Train Epoch : 5 [32000/60000 (53%)]     Train Loss : 0.010567
Train Epoch : 5 [38400/60000 (64%)]     Train Loss : 0.008872
Train Epoch : 5 [44800/60000 (75%)]     Train Loss : 0.008231
Train Epoch : 5 [51200/60000 (85%)]     Train Loss : 0.012911
Train Epoch : 5 [57600/60000 (96%)]     Train Loss : 0.011835

[EPOCH : 5],    TEST LOSS : 0.0003
Train Epoch : 6 [0/60000 (0%)]  Train Loss : 0.009073
Train Epoch : 6 [6400/60000 (11%)]      Train Loss : 0.009686
Train Epoch : 6 [12800/60000 (21%)]     Train Loss : 0.011784
Train Epoch : 6 [19200/60000 (32%)]     Train Loss : 0.011273
Train Epoch : 6 [25600/60000 (43%)]     Train Loss : 0.011015
Train Epoch : 6 [32000/60000 (53%)]     Train Loss : 0.008770
Train Epoch : 6 [38400/60000 (64%)]     Train Loss : 0.009763
Train Epoch : 6 [44800/60000 (75%)]     Train Loss : 0.009289
Train Epoch : 6 [51200/60000 (85%)]     Train Loss : 0.008806
Train Epoch : 6 [57600/60000 (96%)]     Train Loss : 0.009463

[EPOCH : 6],    TEST LOSS : 0.0003
Train Epoch : 7 [0/60000 (0%)]  Train Loss : 0.008830
Train Epoch : 7 [6400/60000 (11%)]      Train Loss : 0.008179
Train Epoch : 7 [12800/60000 (21%)]     Train Loss : 0.010034
Train Epoch : 7 [19200/60000 (32%)]     Train Loss : 0.008470
Train Epoch : 7 [25600/60000 (43%)]     Train Loss : 0.011674
Train Epoch : 7 [32000/60000 (53%)]     Train Loss : 0.009756
Train Epoch : 7 [38400/60000 (64%)]     Train Loss : 0.009217
Train Epoch : 7 [44800/60000 (75%)]     Train Loss : 0.012578
Train Epoch : 7 [51200/60000 (85%)]     Train Loss : 0.009546
Train Epoch : 7 [57600/60000 (96%)]     Train Loss : 0.011373

[EPOCH : 7],    TEST LOSS : 0.0003
Train Epoch : 8 [0/60000 (0%)]  Train Loss : 0.009549
Train Epoch : 8 [6400/60000 (11%)]      Train Loss : 0.009751
Train Epoch : 8 [12800/60000 (21%)]     Train Loss : 0.009334
Train Epoch : 8 [19200/60000 (32%)]     Train Loss : 0.009574
Train Epoch : 8 [25600/60000 (43%)]     Train Loss : 0.009634
Train Epoch : 8 [32000/60000 (53%)]     Train Loss : 0.008717
Train Epoch : 8 [38400/60000 (64%)]     Train Loss : 0.012291
Train Epoch : 8 [44800/60000 (75%)]     Train Loss : 0.012406
Train Epoch : 8 [51200/60000 (85%)]     Train Loss : 0.009570
Train Epoch : 8 [57600/60000 (96%)]     Train Loss : 0.009293

[EPOCH : 8],    TEST LOSS : 0.0003
Train Epoch : 9 [0/60000 (0%)]  Train Loss : 0.012918
Train Epoch : 9 [6400/60000 (11%)]      Train Loss : 0.007843
Train Epoch : 9 [12800/60000 (21%)]     Train Loss : 0.008968
Train Epoch : 9 [19200/60000 (32%)]     Train Loss : 0.011219
Train Epoch : 9 [25600/60000 (43%)]     Train Loss : 0.010170
Train Epoch : 9 [32000/60000 (53%)]     Train Loss : 0.008842
Train Epoch : 9 [38400/60000 (64%)]     Train Loss : 0.008956
Train Epoch : 9 [44800/60000 (75%)]     Train Loss : 0.010373
Train Epoch : 9 [51200/60000 (85%)]     Train Loss : 0.008301
Train Epoch : 9 [57600/60000 (96%)]     Train Loss : 0.009081

[EPOCH : 9],    TEST LOSS : 0.0003
Train Epoch : 10 [0/60000 (0%)] Train Loss : 0.009978
Train Epoch : 10 [6400/60000 (11%)]     Train Loss : 0.009724
Train Epoch : 10 [12800/60000 (21%)]    Train Loss : 0.009630
Train Epoch : 10 [19200/60000 (32%)]    Train Loss : 0.010164
Train Epoch : 10 [25600/60000 (43%)]    Train Loss : 0.010565
Train Epoch : 10 [32000/60000 (53%)]    Train Loss : 0.009277
Train Epoch : 10 [38400/60000 (64%)]    Train Loss : 0.008018
Train Epoch : 10 [44800/60000 (75%)]    Train Loss : 0.008146
Train Epoch : 10 [51200/60000 (85%)]    Train Loss : 0.011406
Train Epoch : 10 [57600/60000 (96%)]    Train Loss : 0.008637

[EPOCH : 10],   TEST LOSS : 0.0003
'''
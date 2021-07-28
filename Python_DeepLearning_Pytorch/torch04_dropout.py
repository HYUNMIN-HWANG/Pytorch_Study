'''1. Module Import'''
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets

'''2. 딥러닝 모델을 설계할 떄 활용하는 장비 확인'''
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else :
    DEVICE = torch.device('cpu')

print("Using PyTorch version : ", torch.__version__, ", Device : ", DEVICE) 
# Using PyTorch version :  1.9.0 , Device :  cuda

BATCH_SIZE = 32
EPOCHS = 10

'''3. MNIST 데이터 다운로드(train, test set 분리하기)'''
train_dataset = datasets.MNIST(
    root = '../data/MNIST',
    train = True,
    download=True,
    transform=transforms.ToTensor()
)

test_dataset = datasets.MNIST(
    root='../data/MNIST',
    train = False,
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
    shuffle=False
)

'''4. 데이터 확인하기(1)'''
for (X_train, y_train) in train_loader :
    print("X_train shape : ", X_train.size(), "type : ", X_train.type())
    print("y_train shape : ", y_train.size(), "type : ", y_train.type())
# X_train shape :  torch.Size([32, 1, 28, 28]) type :  torch.FloatTensor
# y_train shape :  torch.Size([32]) type :  torch.LongTensor

'''5. 데이터 확인하기(2)'''
pltsize=1
plt.figure(figsize=(10*pltsize, pltsize))
for i in range(10) :
    plt.subplot(1, 10, i+1)
    plt.axis('off')
    plt.imshow(X_train[i,:,:,:].numpy().reshape(28,28), cmap="gray_r")
    plt.title("Class : " + str(y_train[i].item()))
plt.show()

'''6. MLP(Multi Layer Perceptron) 모델 설계하기'''
class Net(nn.Module) :
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
        self.dropout_prob = 0.5         # dropout : 몇 퍼센트의 노드에 대해 가중값을 계산하지 않을 것인지를 명시
    
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.fc1(x)
        # x = F.sigmoid(x)
        x = torch.sigmoid(x)
        x = F.dropout(x, training=self.training, p=self.dropout_prob)
                    # x 값에 적용 / 학습 상태일 때 적용 / 몇 퍼센트의 노드에 대해 계산하지 않을 것인지
        x = self.fc2(x)
        # x = F.sigmoid(x)
        x = torch.sigmoid(x)
        x = F.dropout(x, training=self.training, p=self.dropout_prob)
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x

'''7. Optimizer, Objective Funtion'''
model = Net().to(DEVICE)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
criterion = nn.CrossEntropyLoss()

print(model)
# Net(
#   (fc1): Linear(in_features=784, out_features=512, bias=True)
#   (fc2): Linear(in_features=512, out_features=256, bias=True)
#   (fc3): Linear(in_features=256, out_features=10, bias=True)
# )


'''8. MLP 모델 학습을 진행하며 학습 데이터에 대한 모델 성능을 확인하는 함수 정의'''
def train(model, train_loader, optimizer, log_interval):
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
            print("Train Epoch : {} [{}/{}({:.0f}%)]\t Train loss : {:.6f}".format(
                Epoch, batch_idx*len(image), len(train_loader.dataset), 100.*batch_idx / len(train_loader), loss.item()
            ))


'''9. 학습되는 과정 속에서 검증 데이터에 대한 모델 성능을 확인하는 함수 정의'''
def evaluate(model, test_loader) :
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for image, label in test_loader:
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
for Epoch in range(1, EPOCHS+1):
    train(model, train_loader, optimizer, log_interval=200)
    test_loss, test_accuracy = evaluate(model, test_loader)
    print("\n[Epoch : {}], \tTEST LOSS : {:.4f}, \tTEST ACCURACY : {:.2f} %\n".format(
        Epoch, test_loss, test_accuracy
    ))


'''
Train Epoch : 1 [0/60000(0%)]    Train loss : 2.393631
Train Epoch : 1 [6400/60000(11%)]        Train loss : 2.310815
Train Epoch : 1 [12800/60000(21%)]       Train loss : 2.372236
Train Epoch : 1 [19200/60000(32%)]       Train loss : 2.232254
Train Epoch : 1 [25600/60000(43%)]       Train loss : 2.330022
Train Epoch : 1 [32000/60000(53%)]       Train loss : 2.287985
Train Epoch : 1 [38400/60000(64%)]       Train loss : 2.368670
Train Epoch : 1 [44800/60000(75%)]       Train loss : 2.371167
Train Epoch : 1 [51200/60000(85%)]       Train loss : 2.251942
Train Epoch : 1 [57600/60000(96%)]       Train loss : 2.325651

[Epoch : 1],    TEST LOSS : 0.0714,     TEST ACCURACY : 10.09 %

Train Epoch : 2 [0/60000(0%)]    Train loss : 2.271344
Train Epoch : 2 [6400/60000(11%)]        Train loss : 2.323800
Train Epoch : 2 [12800/60000(21%)]       Train loss : 2.342918
Train Epoch : 2 [19200/60000(32%)]       Train loss : 2.244805
Train Epoch : 2 [25600/60000(43%)]       Train loss : 2.252782
Train Epoch : 2 [32000/60000(53%)]       Train loss : 2.247920
Train Epoch : 2 [38400/60000(64%)]       Train loss : 2.273850
Train Epoch : 2 [44800/60000(75%)]       Train loss : 2.262495
Train Epoch : 2 [51200/60000(85%)]       Train loss : 2.197375
Train Epoch : 2 [57600/60000(96%)]       Train loss : 2.156679

[Epoch : 2],    TEST LOSS : 0.0643,     TEST ACCURACY : 36.82 %

Train Epoch : 3 [0/60000(0%)]    Train loss : 2.167336
Train Epoch : 3 [6400/60000(11%)]        Train loss : 2.222014
Train Epoch : 3 [12800/60000(21%)]       Train loss : 2.144808
Train Epoch : 3 [19200/60000(32%)]       Train loss : 1.926223
Train Epoch : 3 [25600/60000(43%)]       Train loss : 1.773158
Train Epoch : 3 [32000/60000(53%)]       Train loss : 1.872247
Train Epoch : 3 [38400/60000(64%)]       Train loss : 1.680596
Train Epoch : 3 [44800/60000(75%)]       Train loss : 1.792937
Train Epoch : 3 [51200/60000(85%)]       Train loss : 1.424894
Train Epoch : 3 [57600/60000(96%)]       Train loss : 1.506979

[Epoch : 3],    TEST LOSS : 0.0389,     TEST ACCURACY : 56.44 %

Train Epoch : 4 [0/60000(0%)]    Train loss : 1.453223
Train Epoch : 4 [6400/60000(11%)]        Train loss : 1.470974
Train Epoch : 4 [12800/60000(21%)]       Train loss : 1.218637
Train Epoch : 4 [19200/60000(32%)]       Train loss : 1.108032
Train Epoch : 4 [25600/60000(43%)]       Train loss : 1.291291
Train Epoch : 4 [32000/60000(53%)]       Train loss : 1.103624
Train Epoch : 4 [38400/60000(64%)]       Train loss : 1.132026
Train Epoch : 4 [44800/60000(75%)]       Train loss : 1.023074
Train Epoch : 4 [51200/60000(85%)]       Train loss : 0.883704
Train Epoch : 4 [57600/60000(96%)]       Train loss : 1.202062

[Epoch : 4],    TEST LOSS : 0.0282,     TEST ACCURACY : 71.10 %

Train Epoch : 5 [0/60000(0%)]    Train loss : 1.231973
Train Epoch : 5 [6400/60000(11%)]        Train loss : 1.487853
Train Epoch : 5 [12800/60000(21%)]       Train loss : 1.177687
Train Epoch : 5 [19200/60000(32%)]       Train loss : 1.071121
Train Epoch : 5 [25600/60000(43%)]       Train loss : 1.040922
Train Epoch : 5 [32000/60000(53%)]       Train loss : 0.929684
Train Epoch : 5 [38400/60000(64%)]       Train loss : 0.882741
Train Epoch : 5 [44800/60000(75%)]       Train loss : 1.028343
Train Epoch : 5 [51200/60000(85%)]       Train loss : 0.944637
Train Epoch : 5 [57600/60000(96%)]       Train loss : 0.816989

[Epoch : 5],    TEST LOSS : 0.0240,     TEST ACCURACY : 75.59 %

Train Epoch : 6 [0/60000(0%)]    Train loss : 0.711574
Train Epoch : 6 [6400/60000(11%)]        Train loss : 1.066163
Train Epoch : 6 [12800/60000(21%)]       Train loss : 0.750500
Train Epoch : 6 [19200/60000(32%)]       Train loss : 0.890703
Train Epoch : 6 [25600/60000(43%)]       Train loss : 0.930133
Train Epoch : 6 [32000/60000(53%)]       Train loss : 0.654478
Train Epoch : 6 [38400/60000(64%)]       Train loss : 0.903774
Train Epoch : 6 [44800/60000(75%)]       Train loss : 0.615565
Train Epoch : 6 [51200/60000(85%)]       Train loss : 0.913747
Train Epoch : 6 [57600/60000(96%)]       Train loss : 0.758832

[Epoch : 6],    TEST LOSS : 0.0208,     TEST ACCURACY : 79.69 %

Train Epoch : 7 [0/60000(0%)]    Train loss : 1.046976
Train Epoch : 7 [6400/60000(11%)]        Train loss : 0.831451
Train Epoch : 7 [12800/60000(21%)]       Train loss : 0.807098
Train Epoch : 7 [19200/60000(32%)]       Train loss : 0.601098
Train Epoch : 7 [25600/60000(43%)]       Train loss : 0.657971
Train Epoch : 7 [32000/60000(53%)]       Train loss : 0.886570
Train Epoch : 7 [38400/60000(64%)]       Train loss : 0.706390
Train Epoch : 7 [44800/60000(75%)]       Train loss : 0.445574
Train Epoch : 7 [51200/60000(85%)]       Train loss : 0.530242
Train Epoch : 7 [57600/60000(96%)]       Train loss : 0.719596

[Epoch : 7],    TEST LOSS : 0.0179,     TEST ACCURACY : 82.87 %

Train Epoch : 8 [0/60000(0%)]    Train loss : 0.508746
Train Epoch : 8 [6400/60000(11%)]        Train loss : 0.650705
Train Epoch : 8 [12800/60000(21%)]       Train loss : 0.810365
Train Epoch : 8 [19200/60000(32%)]       Train loss : 0.844184
Train Epoch : 8 [25600/60000(43%)]       Train loss : 0.609209
Train Epoch : 8 [32000/60000(53%)]       Train loss : 0.724806
Train Epoch : 8 [38400/60000(64%)]       Train loss : 0.465963
Train Epoch : 8 [44800/60000(75%)]       Train loss : 0.829550
Train Epoch : 8 [51200/60000(85%)]       Train loss : 0.634155
Train Epoch : 8 [57600/60000(96%)]       Train loss : 0.853575

[Epoch : 8],    TEST LOSS : 0.0159,     TEST ACCURACY : 84.99 %

Train Epoch : 9 [0/60000(0%)]    Train loss : 0.621066
Train Epoch : 9 [6400/60000(11%)]        Train loss : 1.075188
Train Epoch : 9 [12800/60000(21%)]       Train loss : 0.425928
Train Epoch : 9 [19200/60000(32%)]       Train loss : 0.705092
Train Epoch : 9 [25600/60000(43%)]       Train loss : 0.747972
Train Epoch : 9 [32000/60000(53%)]       Train loss : 0.795489
Train Epoch : 9 [38400/60000(64%)]       Train loss : 0.616677
Train Epoch : 9 [44800/60000(75%)]       Train loss : 0.579610
Train Epoch : 9 [51200/60000(85%)]       Train loss : 0.615294
Train Epoch : 9 [57600/60000(96%)]       Train loss : 0.758350

[Epoch : 9],    TEST LOSS : 0.0147,     TEST ACCURACY : 85.97 %

Train Epoch : 10 [0/60000(0%)]   Train loss : 0.640940
Train Epoch : 10 [6400/60000(11%)]       Train loss : 0.288758
Train Epoch : 10 [12800/60000(21%)]      Train loss : 0.517229
Train Epoch : 10 [19200/60000(32%)]      Train loss : 0.428639
Train Epoch : 10 [25600/60000(43%)]      Train loss : 0.477669
Train Epoch : 10 [32000/60000(53%)]      Train loss : 0.593850
Train Epoch : 10 [38400/60000(64%)]      Train loss : 0.503230
Train Epoch : 10 [44800/60000(75%)]      Train loss : 0.650774
Train Epoch : 10 [51200/60000(85%)]      Train loss : 0.606821
Train Epoch : 10 [57600/60000(96%)]      Train loss : 0.471360

[Epoch : 10],   TEST LOSS : 0.0139,     TEST ACCURACY : 86.82 %
'''
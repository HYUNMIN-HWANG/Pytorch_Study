# optimizer
# batch 단위로 back propagation 하는 과정 
# momentum, NAG, Adagrad, RMSProp, Adadelta, Adam, RAdam

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
def weight_init(m) :                            
    if isinstance(m, nn.Linear) :               
        init.kaiming_uniform_(m.weight.data)    
    
model = NET().to(DEVICE)
model.apply(weight_init)                        
# optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)    # SGD
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Adam
# optimizer = torch.optim.RMSprop(model.parameters(), lr=0.01)  # RMSRrop
# optimizer = torch.optim.Adadelta(model.parameters(), lr=0.01)  # Adadelta
import torch_optimizer as optim
optimizer = optim.RAdam(model.parameters(), lr=0.01)  # RAdam
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
    print("\n[Epoch : {}], \tTRAIN LOSS : {}\tTRAIN ACCURACY : {} %\n".format(
        Epoch, test_loss, test_accuracy
    ))


# SGD
# [Epoch : 10,    TRAIN LOSS : 0.0028271086052962346      TRAIN ACCURACY : 97.21 %

# Adam
# [Epoch : 10,    TRAIN LOSS : 0.0019999692377183236      TRAIN ACCURACY : 98.09 %

# RMSProp
# [Epoch : 10],   TRAIN LOSS : 0.0026743589877421983      TRAIN ACCURACY : 97.72 %

# Adadelta
# [Epoch : 10],   TRAIN LOSS : 0.00600555935963057        TRAIN ACCURACY : 94.24 %

# RAdam
# [Epoch : 10],   TRAIN LOSS : 0.00208582293643758        TRAIN ACCURACY : 98.0 %

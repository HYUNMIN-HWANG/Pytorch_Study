import torch
from torch._C import ParameterDict
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.init

device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(2021)

if device == 'cuda':
    torch.cuda.manual_seed_all(2021)

learning_rate = 0.001
training_epochs = 15
batch_size = 100

'''1. DATA'''
mnist_train = dsets.MNIST(root='D:\\Data\\MNIST_data\\',
                          train = True,     # 훈련 데이터로 지정
                          transform=transforms.ToTensor(),  # 텐서로 변환
                          download=True)

mnist_test = dsets.MNIST(root='D:\\Data\\MNIST_data\\',
                          train = False,     # 훈련 데이터로 지정
                          transform=transforms.ToTensor(),  # 텐서로 변환
                          download=True)

data_loader = torch.utils.data.DataLoader(dataset=mnist_train,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          drop_last=True)   # drops the last non-full batch of each worker’s iterable-style dataset replica

'''2. Model'''
class CNN(torch.nn.Module) :

    def __init__(self) :
        super(CNN, self).__init__()
        self.keep_prob = 0.5    # 주어진 유닛을 유지할 확률 즉, drop하지 않을 확률

        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2,  padding=1)
        )

        self.fc1 = torch.nn.Linear(4*4*128, 625, bias=True)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.layer4 = torch.nn.Sequential(
            self.fc1, 
            torch.nn.ReLU(),
            torch.nn.Dropout(p=1-self.keep_prob)
        )

        self.fc2 = torch.nn.Linear(625, 10, bias=True)
        torch.nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x) :
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        flat = x.view(x.size(0), -1)
        out = self.layer4(flat)
        out = self.fc2(out)
        return out


model = CNN().to(device)

criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

total_batch = len(data_loader)
print(total_batch)  # 600

'''3. Train'''
for epoch in range(training_epochs) :
    avg_cost = 0

    for X, y in data_loader :
        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        hypothesis = model(X)
        cost = criterion(hypothesis, y)
        cost.backward()
        optimizer.step()

        avg_cost += cost / total_batch
    
    print("Epoch : {:4d} | cost = {:8f}".format(epoch+1, avg_cost))


'''4. Test'''
with torch.no_grad() :
    X_test = mnist_test.test_data.view(len(mnist_test), 1, 28, 28).float().to(device)
    y_test = mnist_test.test_labels.to(device)

    prediction = model(X_test)

    correct_prediction = torch.argmax(prediction, 1) == y_test
    accuracy = correct_prediction.float().mean()
    print("Accuracy : ", accuracy)


"""
Epoch :    1 | cost = 0.190107
Epoch :    2 | cost = 0.047783
Epoch :    3 | cost = 0.035162
Epoch :    4 | cost = 0.028263
Epoch :    5 | cost = 0.021998
Epoch :    6 | cost = 0.018854
Epoch :    7 | cost = 0.017275
Epoch :    8 | cost = 0.013697
Epoch :    9 | cost = 0.013000
Epoch :   10 | cost = 0.009658
Epoch :   11 | cost = 0.010600
Epoch :   12 | cost = 0.008914
Epoch :   13 | cost = 0.010678
Epoch :   14 | cost = 0.006895
Epoch :   15 | cost = 0.006538

Accuracy :  tensor(0.9774, device='cuda:0')
"""
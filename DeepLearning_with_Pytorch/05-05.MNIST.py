import torch
from torch.optim import optimizer
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, dataloader
import torch.nn as nn
import matplotlib.pyplot as plt
import random

USE_CUDA = torch.cuda.is_available()    # GPU 사용 가능하면 True 반환
device = torch.device("cuda" if USE_CUDA else "cpu")
print(device)

random.seed(2021)
torch.manual_seed(2021)
if device == "cuda" :
    torch.cuda.manual_seed_all(2021)

training_epoch = 15
batch_size = 32

'''DATA'''
mnist_train = dsets.MNIST(root='D:\Data\MNIST_data/', # 다운로드 경로 지정
                          train=True, # True를 지정하면 훈련 데이터로 다운로드
                          transform=transforms.ToTensor(), # 텐서로 변환
                          download=True)

mnist_test = dsets.MNIST(root='D:\Data\MNIST_data/', # 다운로드 경로 지정
                         train=False, # False를 지정하면 테스트 데이터로 다운로드
                         transform=transforms.ToTensor(), # 텐서로 변환
                         download=True)

data_loader = DataLoader(dataset=mnist_train,
                         batch_size=batch_size,
                         shuffle=True,
                         drop_last=True) # drop_last=True : 마지막에 남은 배치를 버린다. 

'''Model'''
# 28*28 = 784
linear = nn.Linear(784, 10, bias=True).to(device)

criterion = nn.CrossEntropyLoss().to(device)    # nn.CrossEntropyLoss() == torch.nn.functional.cross_entropy()
optimizier = torch.optim.SGD(linear.parameters(), lr=0.01)

'''Train'''
for epoch in range(training_epoch):
    avg_cost = 0
    total_batch = len(data_loader)

    for X, Y in data_loader:
        X = X.view(-1, 28*28).to(device)
        Y = Y.to(device)

        hypothesis = linear(X)
        cost = criterion(hypothesis, Y)

        optimizier.zero_grad()
        cost.backward()
        optimizier.step()

        avg_cost += cost / total_batch

    print("Epoch : ", "%04d" % (epoch+1), "cost = ", "{:.9f}".format(avg_cost))

print("The end")

# Epoch :  0001 cost =  0.767483652
# Epoch :  0002 cost =  0.453228772
# Epoch :  0003 cost =  0.401512951
# Epoch :  0004 cost =  0.375414938
# Epoch :  0005 cost =  0.358982921
# Epoch :  0006 cost =  0.347260833
# Epoch :  0007 cost =  0.338285059
# Epoch :  0008 cost =  0.331262469
# Epoch :  0009 cost =  0.325454235
# Epoch :  0010 cost =  0.320703357
# Epoch :  0011 cost =  0.316435486
# Epoch :  0012 cost =  0.312887967
# Epoch :  0013 cost =  0.309693307
# Epoch :  0014 cost =  0.306827068
# Epoch :  0015 cost =  0.304248571

'''Prediction'''
with torch.no_grad():
    X_test = mnist_test.test_data.view(-1, 28*28).float().to(device)
    Y_test = mnist_test.test_labels.to(device)
    print(X_test.shape)
    print(Y_test.shape)
    # torch.Size([10000, 784])
    # torch.Size([10000])

    prediction = linear(X_test)
    correct_prediction = torch.argmax(prediction, 1) == Y_test
    accuracy = correct_prediction.float().mean()
    print("Accuracy : ", accuracy.item())
    # Accuracy :  0.902899980545044

    # MNIST 테스트 데이터에서 무작위로 하나 뽑아서 예측
    r = random.randint(0, len(mnist_test)-1)
    x_single_data = mnist_test.test_data[r:r+1].view(-1, 28*28).float().to(device)
    y_single_data = mnist_test.test_labels[r:r+1].to(device)

    print("Label : ", y_single_data.item())
    single_prediction = linear(x_single_data)
    print("prediction : ", torch.argmax(single_prediction,1).item())

    plt.imshow(mnist_test.test_data[r:r+1].view(28,28), cmap="gray", interpolation="nearest")
    plt.show()
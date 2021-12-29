# 다층 퍼셉트론 구현하기
import torch
from torch.autograd.grad_mode import no_grad
import torch.nn as nn

device = 'cpu'if torch.cuda.is_available() else 'cpu'

torch.manual_seed(2021)
torch.cuda.manual_seed_all(2021)

X = torch.FloatTensor([[0,0], [0,1], [1,0], [1,1]]).to(device)
Y = torch.FloatTensor([[0], [1], [1], [0]]).to(device)

# 다층
model = nn.Sequential(
    nn.Linear(2, 10, bias=True),
    nn.Sigmoid(),
    nn.Linear(10, 10, bias=True),
    nn.Sigmoid(),
    nn.Linear(10, 10, bias=True),
    nn.Sigmoid(),
    nn.Linear(10, 1, bias=True),
    nn.Sigmoid()
).to(device)


criterion = torch.nn.BCELoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=1)  


for epoch in range(10001) :
    optimizer.zero_grad()
    
    # forward 연산
    hypothesis = model(X)
    
    # cost 함수
    cost = criterion(hypothesis, Y)
    cost.backward()
    optimizer.step()
    
    if epoch % 100 == 0 :
        print(epoch, cost.item())
"""
0 0.7753092646598816
100 0.6931654214859009
200 0.6931637525558472
300 0.693162202835083
400 0.6931605339050293
500 0.6931591033935547
600 0.6931576728820801
700 0.6931563019752502
800 0.6931549310684204
900 0.6931535601615906
1000 0.6931522488594055"""

with torch.no_grad():
    hypothesis = model(X)
    predicted = (hypothesis > 0.5).float()
    accuracy = (predicted == Y).float().mean()
    print("모델의 출력값 Hypothesis : \n", hypothesis.detach().cpu().numpy())
    print("모델의 예측값 Predictd : \n", predicted.detach().cpu().numpy())    
    print("실제값 (Y) : \n", Y.cpu().numpy())
    print("정확도 Accuracy : \n", accuracy.item())
"""
모델의 출력값 Hypothesis :
 [[1.7748027e-04]
 [9.9982589e-01]
 [9.9984074e-01]
 [1.6117562e-04]]
모델의 예측값 Predictd :
 [[0.]
 [1.]
 [1.]
 [0.]]
실제값 (Y) :
 [[0.]
 [1.]
 [1.]
 [0.]]
정확도 Accuracy :
 1.0
"""
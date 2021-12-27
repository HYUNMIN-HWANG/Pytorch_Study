# 단층 퍼셉트론 구현하기
import torch
import torch.nn as nn
from torch.nn.functional import sigmoid


device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(2022)
if device == 'cuda' :
    torch.cuda.manual_seed(2022)
    
X = torch.FloatTensor([[0,0],[0,1],[1,0],[1,1]]).to(device)
Y = torch.FloatTensor([[0],[1],[1],[0]]).to(device)

linear = nn.Linear(2, 1, bias=True)
sigmoid = nn.Sigmoid()
model = nn.Sequential(linear, sigmoid).to(device)

criterion = torch.nn.BCELoss().to(device)       # 이진 분류에서 사용하는 로스 함수
optimizer = torch.optim.SGD(model.parameters(), lr=1)

for step in range(1001) :
    optimizer.zero_grad()
    hypothesis = model(X)
    
    # Loss
    cost = criterion(hypothesis, Y)
    cost.backward()
    optimizer.step()
    
    if step % 100 == 0 :
        print(step, cost.item())

"""
0 0.7469314932823181
100 0.6931471824645996
200 0.6931471824645996
300 0.6931471824645996
400 0.6931471824645996
500 0.6931471824645996
600 0.6931471824645996
700 0.6931471824645996
800 0.6931471824645996
900 0.6931471824645996
1000 0.6931471824645996
"""

with torch.no_grad() :
    hypothesis = model(X)
    predicted = (hypothesis > 0.5).float()
    accuracy = (predicted == Y).float().mean()
    print("모델의 출력값 hypothesis : \n", hypothesis.detach().cpu().numpy())
    print("모델의 예측값 predictied : \n", predicted.detach().cpu().numpy())
    print("실제 값 Y : \n", Y.cpu().numpy())
    print("정확도 Accuracy : ", accuracy.item())

"""
모델의 출력값 hypothesis :
 [[0.49991977]
 [0.5001145 ]
 [0.49988547]
 [0.5000803 ]]
모델의 예측값 predictied :
 [[0.]
 [1.]
 [0.]
 [1.]]
실제 값 Y :
 [[0.]
 [1.]
 [1.]
 [0.]]
정확도 Accuracy :  0.5

=> 제대로 예측하지 못함

"""
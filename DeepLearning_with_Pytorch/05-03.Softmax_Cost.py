import torch
import torch.nn.functional as F

torch.manual_seed(1)

'''softmax 비용 함수 구현하기 (low level)'''
# 벡터
    # 클래스가 3개인 경우
z = torch.FloatTensor([1,2,3])

hypothesis = F.softmax(z, dim=0)
print(hypothesis)   # tensor([0.0900, 0.2447, 0.6652])
print(hypothesis.sum()) # tensor(1.)  : softmax 모든 원소들의 합이 1이다.

# 행렬
    # 클래스가 5개인 경우
    # 3개 샘플에 대해서 5개의 클래스 중 어떤 클래스가 정답인가
z = torch.rand(3, 5, requires_grad=True)

## 1. softmax
hypothesis = F.softmax(z, dim=1)
print(hypothesis)
# tensor([[0.2645, 0.1639, 0.1855, 0.2585, 0.1277],
#         [0.2430, 0.1624, 0.2322, 0.1930, 0.1694],
#         [0.2226, 0.1986, 0.2326, 0.1594, 0.1868]], grad_fn=<SoftmaxBackward>)
print(hypothesis.sum())
# tensor(3.0000, grad_fn=<SumBackward0>)

# 각 샘플에 대한 임의의 레이블을 만든다.
y = torch.randint(5, (3,)).long()
print(y)    # tensor([0, 2, 1])

y_one_hot = torch.zeros_like(hypothesis)    # 모든 원소가 0의 값을 갖는 3X5 행렬을 만든다.

print(y.unsqueeze(1))   # y.unsqueeze(1) : 3크기를 -> (3X1) 텐서가 된다
# tensor([[0],
#         [2],
#         [1]])

y_one_hot.scatter_(1, y.unsqueeze(1), 1)    # scatter (dim = 1, 텐서, 텐서 값이 있는 위치에 1을 넣는다.)
print(y_one_hot)
# tensor([[1., 0., 0., 0., 0.],
#         [0., 0., 1., 0., 0.],
#         [0., 1., 0., 0., 0.]])

## 2. log_softmax
cost = (y_one_hot * -torch.log(hypothesis)).sum(dim=1).mean()
print(cost) # tensor(1.4689, grad_fn=<MeanBackward0>)

'''softmax 비용 함수 구현하기 (high level)'''
hypothesis = F.log_softmax(z, dim=1)    # log_softmax : softmax() + log()
print(hypothesis)
# tensor([[-1.3301, -1.8084, -1.6846, -1.3530, -2.0584],
#         [-1.4147, -1.8174, -1.4602, -1.6450, -1.7758],
#         [-1.5025, -1.6165, -1.4586, -1.8360, -1.6776]],
#        grad_fn=<LogSoftmaxBackward>)

cost = (y_one_hot * -F.log_softmax(z, dim=1)).sum(dim=1).mean()
print(cost) # tensor(1.4689, grad_fn=<MeanBackward0>

## 3. nll_loss
# nll_loss : 원핫벡터를 넣을 필요없이 실제값을 인자로 사용한다.
# nll : Negative Log Likelihood
hypothesis = F.nll_loss(F.log_softmax(z, dim=1), y)
print(hypothesis)   # tensor(1.4689, grad_fn=<NllLossBackward>)

## 4. cross entropy : log_softmax & nll_loss
cost = F.cross_entropy(z, y)
print(cost) # tensor(1.4689, grad_fn=<NllLossBackward>)
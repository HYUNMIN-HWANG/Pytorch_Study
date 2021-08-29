### 벡터 / 행렬 / 텐서 ###
'''
벡터 : 1차원으로 구성된 값 (= 1d tensor)
행렬 : 2차원으로 구성된 값 (= 2d tensor)
    - 행 Batch size X 열 dimension
텐서 : 3차원으로 구성된 값
    - 비전 : 세로 Batch Size X 가로 Width X 높이 Height
    - NLP : Batch Size X 문장의 길이 Length X 단어 벡터 차원 Dimension
'''

### Numpy ###
import numpy as np

'''1D'''
t = np.array([0., 1., 2., 3., 4., 5., 6.])
print(t)
# [0. 1. 2. 3. 4. 5. 6.]

print('Rank of t : ', t.ndim)
print('Shape of t : ', t.shape)
# Rank of t :  1    
# Shape of t :  (7,)  <- (1,7)과 동일한 뜻

print("t[0] t[1] t[-1] : ", t[0], t[1], t[-1])
# t[0] t[1] t[-1] :  0.0 1.0 6.0

print("t[2:5] t[4:-1] : ", t[2:5], t[4:-1])
# t[2:5] t[4:-1] :  [2. 3. 4.] [4. 5.]

print("t[2:] t[:3] : ",t[2:], t[:3] )
# t[2:] t[:3] :  [2. 3. 4. 5. 6.] [0. 1. 2.]

'''2D'''
t = np.array([[1.,2.,3.],[4.,5.,6.],[7., 8., 9.],[10., 11., 12.]])
print(t)
# [[ 1.  2.  3.]
#  [ 4.  5.  6.]
#  [ 7.  8.  9.]
#  [10. 11. 12.]]

print('Rank of t : ', t.ndim)
print('Shape of t : ', t.shape)
# Rank of t :  2
# Shape of t :  (4, 3)

### Pytorch ###
import torch

'''1D'''
t = torch.FloatTensor([0., 1., 2., 3., 4., 5., 6.])
print(t)
# tensor([0., 1., 2., 3., 4., 5., 6.])

print(t.dim())
print(t.shape)
print(t.size())
# 1
# torch.Size([7])
# torch.Size([7])

print(t[0], t[1], t[-1])
print(t[2:5], t[4:-1])
print(t[2:], t[:3])
# tensor(0.) tensor(1.) tensor(6.)
# tensor([2., 3., 4.]) tensor([4., 5.])
# tensor([2., 3., 4., 5., 6.]) tensor([0., 1., 2.])

'''2D'''
t = torch.FloatTensor([[1., 2., 3.],
                        [4., 5., 6.],
                        [7., 8., 9.],
                        [10., 11., 12.]])
print(t)
# tensor([[ 1.,  2.,  3.],
#         [ 4.,  5.,  6.],
#         [ 7.,  8.,  9.],
#         [10., 11., 12.]])

print(t.dim())
print(t.shape)
print(t.size())
# 2
# torch.Size([4, 3])
# torch.Size([4, 3])

print(t[:,1])
print(t[:,1].size())
# tensor([ 2.,  5.,  8., 11.])
# torch.Size([4])

print(t[:,:-1])
# tensor([[ 1.,  2.],
#         [ 4.,  5.],
#         [ 7.,  8.],
#         [10., 11.]])

'''BroadCasting'''
# 크기가 다른 행렬의 크기를 자동으로 맞추어 연산을 수행하게 만든다.

# 크기가 같을 때 (1,2)
m1 = torch.FloatTensor([[3,3]])
m2 = torch.FloatTensor([[2,2]])
print(m1+m2)
# tensor([[5., 5.]])

# 크기가 다를 때
m1 = torch.FloatTensor([[1,2]]) # (1,2)
m2 = torch.FloatTensor([3])     # (1,)  [3] -> [3,3]으로 바꿔서 계산함
print(m1+m2)
# tensor([[4., 5.]])

m1 = torch.FloatTensor([[1, 2]])    # (1,2) -> (2,2)
m2 = torch.FloatTensor([[3], [4]])  # (2,1) -> (2,2)로 바꿔서 계산함
print(m1 + m2)
# tensor([[4., 5.],
#         [5., 6.]])

### Matrix Multiplication ###
'''matmul 행렬곱'''
m1 = torch.FloatTensor([[1,2],[3,4]])
m2 = torch.FloatTensor([[1],[2]])
print(m1.shape, m2.shape)
# torch.Size([2, 2]) torch.Size([2, 1])
print(m1.matmul(m2))
# tensor([[ 5.],
#         [11.]])

'''mul 원소 별 곱'''
# 동일한 위치에 있는 원소끼리 곱셈
m1 = torch.FloatTensor([[1,2],[3,4]])
m2 = torch.FloatTensor([[1],[2]])       # [1], [2] -> [[1],[1]], [[2],[2]]
print(m1.mul(m2))
# tensor([[1., 2.],
#         [6., 8.]])
print(m1 * m2)
# tensor([[1., 2.],
#         [6., 8.]])

### Others ###
'''mean'''
t = torch.FloatTensor([1,2])
print(t.mean())
# tensor(1.5000)

t = torch.FloatTensor([[1,2],[3,4]])    # (2,2)
print(t.mean())
# tensor(2.5000)

print(t.mean(dim=0))    # (1,2) dim=0 첫번째 차원, 행 -> 해당 차원을 제거한다. -> 열만 남긴다. 열에 따른 평균을 구한다.
# tensor([2., 3.])

print(t.mean(dim=1))    # (2,1) 열이 제거된 차원 -> 결국 1차원임 (1,2)과 동일한 표기 가능
# tensor([1.5000, 3.5000])

print(t.mean(dim=-1))   # 마지막 차원을 제거 == 열을 제거 
# tensor([1.5000, 3.5000])

'''Sum'''
t = torch.FloatTensor([[1,2],[3,4]])
print(t.sum())
# tensor(10.)
print(t.sum(dim=0)) # 행 제거
# tensor([4., 6.])
print(t.sum(dim=1)) # 열 제거
# tensor([3., 7.])
print(t.sum(dim=-1))    # 마지막 차원 = 열 제거
# tensor([3., 7.])

'''Max'''
t = torch.FloatTensor([[1,2],[3,4]])
print(t.max())
# tensor(4.)
print(t.max(dim=0)) # 행 제거 -> max 값 뿐만 아니라 argmax 값(인덱스)도 같이 리턴한다.
# torch.return_types.max(
# values=tensor([3., 4.]),
# indices=tensor([1, 1]))

print("Max : ", t.max(dim=0)[0])
print("Argmax : ", t.max(dim=0)[1])
# Max :  tensor([3., 4.])
# Argmax :  tensor([1, 1])

print(t.max(dim=1)) # 열 제거
# torch.return_types.max(
# values=tensor([2., 4.]),
# indices=tensor([1, 1]))

print(t.max(dim=-1))
# torch.return_types.max(
# values=tensor([2., 4.]),
# indices=tensor([1, 1]))

import numpy as np
from numpy.core.arrayprint import printoptions
import torch 

### View ###
# View : 넘파이에서 reshape와 같은 역할을 한다.
    # view는 변경 전과 변경 후의 텐서 안의 원소의 개수가 유지되어야 한다.
    # view에서 사이즈가 -1로 설정되면 다른 차원으로부터 해당 값을 유추합니다.

t = np.array([[[0, 1, 2],
               [3, 4, 5]],
              [[6, 7, 8],
               [9, 10, 1]]])

ft = torch.FloatTensor(t)

print(ft.shape)
# torch.Size([2, 2, 3])

'''3d -> 2d'''
print(ft.view([-1, 3]))         # 두 번째 차원의 길이는 3으로 정한다.
print(ft.view([-1, 3]).shape)
# tensor([[ 0.,  1.,  2.],
#         [ 3.,  4.,  5.],
#         [ 6.,  7.,  8.],
#         [ 9., 10.,  1.]])
# torch.Size([4, 3])

'''3d -> 3d shape 변경'''
print(ft.view([-1, 1, 3]))
print(ft.view([-1, 1, 3]).shape)
# tensor([[[ 0.,  1.,  2.]],

#         [[ 3.,  4.,  5.]],

#         [[ 6.,  7.,  8.]],

#         [[ 9., 10.,  1.]]])
# torch.Size([4, 1, 3])

### Squeeze ###
# 차원이 1인 경우의 해당 차원을 제거

ft = torch.FloatTensor([[0], [1], [2]])
print(ft)
print(ft.shape)
# tensor([[0.],
#         [1.],
#         [2.]])
# torch.Size([3, 1])

print(ft.squeeze())
print(ft.squeeze().shape)
# tensor([0., 1., 2.])
# torch.Size([3])

### Unsqueeze ###
# 특정 위치에 1인 차원을 추가할 수 있다.
ft = torch.FloatTensor([0, 1, 2])
print(ft.shape)
# torch.Size([3])

print(ft.unsqueeze(0))  # 첫 번째 차원에 1인 차원이 추가됨
print(ft.unsqueeze(0).shape)
# tensor([[0., 1., 2.]])
# torch.Size([1, 3])

print(ft.view([1, -1])) # view로도 구현할 수 있다.
print(ft.view([1, -1]).shape)
# tensor([[0., 1., 2.]])
# torch.Size([1, 3])

print(ft.unsqueeze(1))  # 두 번째 차원에 1인 차원이 추가됨
print(ft.unsqueeze(1).shape)
# tensor([[0.],
#         [1.],
#         [2.]])
# torch.Size([3, 1])

print(ft.unsqueeze(-1))  # 마지막 차원에 1인 차원이 추가됨
print(ft.unsqueeze(-1).shape)
# tensor([[0.],
#         [1.],
#         [2.]])
# torch.Size([3, 1])

### Type Casting ###
# 자료형을 변환하는 것

lt = torch.LongTensor([1, 2, 3, 4])
print(lt)
# tensor([1, 2, 3, 4])

print(lt.float())   # float형으로 타입 변경
# tensor([1., 2., 3., 4.])

bt = torch.ByteTensor([True, False, False, True])
print(bt)
# tensor([1, 0, 0, 1], dtype=torch.uint8)

print(bt.long())    # long 타입의 텐서로 변경
# tensor([1, 0, 0, 1])
print(bt.float())   # float 타입의 텐서로 변경
# tensor([1., 0., 0., 1.])

### Concatenate ###
x = torch.FloatTensor([[1,2], [3,4]])   # (2,2)
y = torch.FloatTensor([[5,6], [7,8]])   # (2,2)

print(torch.cat([x, y], dim=0)) # dim=0 : 첫 번째 차원을 늘린다.
# tensor([[1., 2.],
#         [3., 4.],
#         [5., 6.],
#         [7., 8.]])    # (4,2)

print(torch.cat([x, y], dim=1)) # dim=1 : 두 번째 차원을 늘린다.
# tensor([[1., 2., 5., 6.], 
#         [3., 4., 7., 8.]])    # (2,4)



### Stacking ###
x = torch.FloatTensor([1, 4])
y = torch.FloatTensor([2, 5])
z = torch.FloatTensor([3, 6])

print(torch.stack([x, y, z]))
# tensor([[1., 4.],
#         [2., 5.],
#         [3., 6.]])

print(torch.stack([x, y, z], dim=0))
# tensor([[1., 4.],
#         [2., 5.],
#         [3., 6.]])

print(torch.stack([x, y, z], dim=1))
# tensor([[1., 2., 3.],
#         [4., 5., 6.]])

### ones_like , zeros_like ###
x = torch.FloatTensor([[0, 1, 2], [2, 1, 0]])
print(x)
# tensor([[0., 1., 2.],
#         [2., 1., 0.]])

print(torch.ones_like(x))   #  ones_like  : 1로만 채워진 텐서 생성
# tensor([[1., 1., 1.],
#         [1., 1., 1.]])

print(torch.zeros_like(x))  # zeros_like : 0으로만 채워진 텐서 생성
# tensor([[0., 0., 0.],
#         [0., 0., 0.]])

### in-place operation ###
# 덮어쓰기 연산

x = torch.FloatTensor([[1, 2], [3, 4]])
print(x.mul(2.))
print(x)    # 2를 곱해도 x는 변하지 않는다.
# tensor([[2., 4.],
#         [6., 8.]])
# tensor([[1., 2.],
#         [3., 4.]])

print(x.mul_(2.))   # mul_을 하면 기존의 x 값 덮어쓴다.
print(x)
# tensor([[2., 4.],
#         [6., 8.]])
# tensor([[2., 4.],
#         [6., 8.]])


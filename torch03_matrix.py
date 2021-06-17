# [3] Matrix 행렬
# 2개 이상의 벡터 값을 통합해 구성된 값
# 벡터 값 간 연산 속도를 빠르게 진행할 수 있는 선형 대수의 기본 단위

import torch

matrix1 = torch.tensor([[1., 2.], [3., 4.]])
print(matrix1)
# tensor([[1., 2.],
#         [3., 4.]])

matrix2 = torch.tensor([[5., 6.], [7., 8.]])
print(matrix2)
# tensor([[5., 6.],
#         [7., 8.]])

# 사칙연산 1 
add_matrix = matrix1 + matrix2
sub_matrix = matrix1 - matrix2
mul_matrix = matrix1 * matrix2
div_matrix = matrix1 / matrix2

print("Add : ", add_matrix)             
# Add :  tensor([[ 6.,  8.],
#         [10., 12.]])
print("Subtraction : ", sub_matrix)     
# Subtraction :  tensor([[-4., -4.],
#         [-4., -4.]])
print("Multiply : ", mul_matrix)        
# Multiply :  tensor([[ 5., 12.],
#         [21., 32.]])
print("Divide : ", div_matrix)          
# Divide :  tensor([[0.2000, 0.3333],
#         [0.4286, 0.5000]])


# 사칙연산 2
print("Add: ", torch.add(matrix1, matrix2))             
# Add:  tensor([[ 6.,  8.],
#         [10., 12.]])
print("Subtraction : ", torch.sub(matrix1, matrix2))     
# Subtraction :  tensor([[-4., -4.],
#         [-4., -4.]])
print("Multiply : ", torch.mul(matrix1, matrix2))        
# Multiply :  tensor([[ 5., 12.],
#         [21., 32.]])
print("Divide : ", torch.div(matrix1, matrix2))         
# Divide :  tensor([[0.2000, 0.3333],
#         [0.4286, 0.5000]])

# 행렬 곱 연산
print("Matmul : ", torch.matmul(matrix1, matrix2))
# Matmul :  tensor([[19., 22.],
#         [43., 50.]])
# (1 , 2      (5 , 6     (1*5 + 2*7 , 1*6 + 2*8
#         x          = 
#  3 , 4)      7 , 8)     3*5 + 4*7 , 3*6 + 4*8)

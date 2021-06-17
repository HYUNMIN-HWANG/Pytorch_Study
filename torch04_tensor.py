# [4] Tensor 텐서
# 2차원 이상의 배열

import torch

tensor1 = torch.tensor([[[1., 2.],[3., 4.]], [[5., 6.], [7., 8.]]])
print(tensor1)
# tensor([[[1., 2.],
#          [3., 4.]],

#         [[5., 6.],
#          [7., 8.]]])

tensor2 = torch.tensor([[[9., 10.], [11., 12.]], [[13., 14.], [15., 16.]]])
print(tensor2)
# tensor([[[ 9., 10.],
#          [11., 12.]],

#         [[13., 14.],
#          [15., 16.]]])

# 사칙연산 1 
add_tensor = tensor1 + tensor2
sub_tensor = tensor1 - tensor2
mul_tensor = tensor1 * tensor2
div_tensor = tensor1 / tensor2

print("Add : ", add_tensor)             
# Add :  tensor([[[10., 12.],
#          [14., 16.]],

#         [[18., 20.],
#          [22., 24.]]])
print("Subtraction : ", sub_tensor)     
# Subtraction :  tensor([[[-8., -8.],
#          [-8., -8.]],

#         [[-8., -8.],
#          [-8., -8.]]])
print("Multiply : ", mul_tensor)        
# Multiply :  tensor([[[  9.,  20.],
#          [ 33.,  48.]],

#         [[ 65.,  84.],
#          [105., 128.]]])
print("Divide : ", div_tensor)          
# Divide :  tensor([[[0.1111, 0.2000],
#          [0.2727, 0.3333]],

#         [[0.3846, 0.4286],
#          [0.4667, 0.5000]]])


# 사칙연산 2
print("Add: ", torch.add(tensor1, tensor2))             
# Add:  tensor([[[10., 12.],
#          [14., 16.]],

#         [[18., 20.],
#          [22., 24.]]])
print("Subtraction : ", torch.sub(tensor1, tensor2))     
# Subtraction :  tensor([[[-8., -8.],
#          [-8., -8.]],

#         [[-8., -8.],
#          [-8., -8.]]])
print("Multiply : ", torch.mul(tensor1, tensor2))        
# Multiply :  tensor([[[  9.,  20.],
#          [ 33.,  48.]],

#         [[ 65.,  84.],
#          [105., 128.]]])
print("Divide : ", torch.div(tensor1, tensor2))         
# Divide :  tensor([[[0.1111, 0.2000],
#          [0.2727, 0.3333]],

#         [[0.3846, 0.4286],
#          [0.4667, 0.5000]]])

# 행렬 곱 연산
print("Matmul : ", torch.matmul(tensor1, tensor2))
# Matmul :  tensor([[[ 31.,  34.],
#          [ 71.,  78.]],

#         [[155., 166.],
#          [211., 226.]]])

# ([1, 2], [5, 6]        ([9, 10], [13, 14]       ( [(1*9 + 2*11) , (1*10 + 2*12)], [(5*13 + 6*15) , (5*14 + 6*16)]
#                     x                        = 
#  [3, 4], [7, 8])        [11, 12], [15, 16])       [(3*9 + 4*11) , (3*10 + 4*12)], [(7*13 + 8*15) , (7*14 + 8*16)])

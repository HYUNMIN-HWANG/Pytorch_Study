# [2] Vector 벡터
# 하나의 값을 표현할 때 2개 이상의 수치로 표현한 것
# 여러 수치 값을 이요해 표현하는 방식

import torch

vector1 = torch.tensor([1., 2., 3.])
print(vector1)  # tensor([1., 2., 3.])

vector2 = torch.tensor([4., 5., 6.])
print(vector2)  # tensor([4., 5., 6.])


# 사칙연산 1
add_vector = vector1 + vector2
print(add_vector)   # tensor([5., 7., 9.])

sub_vector = vector1 - vector2
print(sub_vector)   # tensor([-3., -3., -3.])

mul_vector = vector1 * vector2
print(mul_vector)   # tensor([ 4., 10., 18.])

div_vector = vector1 / vector2
print(div_vector)   # tensor([0.2500, 0.4000, 0.5000])


# 사칙연산 2
print("Add: ", torch.add(vector1, vector2))              # Add :  tensor([5., 7., 9.])
print("Subtraction : ", torch.sub(vector1, vector2))     # Subtraction :  tensor([-3., -3., -3.])
print("Multiply : ", torch.mul(vector1, vector2))        # Multiply :  tensor([ 4., 10., 18.])
print("Divide : ", torch.div(vector1, vector2))          # Divide :  tensor([0.2500, 0.4000, 0.5000])
print("Dot : ", torch.dot(vector1, vector2))             # Dot :  tensor(32.)  << 벡터 값 간 내적 연산 : (1*4) + (2*5) + (3*6) = 32

# tensor (텐서)
# 데이터를 표현하는 단위

# [1] Scalar
#상수값, 하나의 값을 표현할 때 1개의 수치로 표현한 것

import torch

scalar1 = torch.tensor([1.])
print(scalar1)
# tensor([1.])

scalar2 = torch.tensor([3.])
print(scalar2)
# tensor([3.])

# 사칙연산 1 
add_scalar = scalar1 + scalar2
sub_scalar = scalar1 - scalar2
mul_scalar = scalar1 * scalar2
div_scalar = scalar1 / scalar2

print("Add : ", add_scalar)             # Add :  tensor([4.])
print("Subtraction : ", sub_scalar)     # Subtraction :  tensor([-2.])
print("Multiply : ", mul_scalar)        # Multiply :  tensor([3.])
print("Divide : ", div_scalar)          # Divide :  tensor([0.3333])


# 사칙연산 2
print("Add: ", torch.add(scalar1, scalar2))              # Add :  tensor([4.])
print("Subtraction : ", torch.sub(scalar1, scalar2))     # Subtraction :  tensor([-2.])
print("Multiply : ", torch.mul(scalar1, scalar2))        # Multiply :  tensor([3.])
print("Divide : ", torch.div(scalar1, scalar2))          # Divide :  tensor([0.3333])
# 자동 미분 (Autograd)
    # requires_grad = True
    # backward()

import torch

w = torch.tensor(2.0, requires_grad = True)   # requires_grad = True : 텐서에 대한 기울기를 저장하겠다. 나중에 grad하면 w에 대해서 미분을 할 수 있다.

y = w ** 2 
z = 2 * y + 5

# 해당 수식의 w에 대한 기울기를 계산함
z.backward()

print("수식을 w로 미분한 값 : {}".format(w.grad)) # w에 대해서 미분
# 수식을 w로 미분한 값 : 8.0

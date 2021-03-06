# 파이토치는 미분을 통해 얻은 기울기를 이전에 계산된 기울기 값에 누적시키는 특징이 있다.
# 따라서 기울기를 계속 0으로 초기화시켜줘야 한다.

import torch
w = torch.tensor(2.0, requires_grad=True)

epochs = 20
for e in range(epochs+1) : 
    z = 2 * w
    
    z.backward()

    print("수식을 w로 미분한 값 : {}".format(w.grad))

# 수식을 w로 미분한 값 : 2.0
# 수식을 w로 미분한 값 : 4.0
# 수식을 w로 미분한 값 : 6.0
# 수식을 w로 미분한 값 : 8.0
# 수식을 w로 미분한 값 : 10.0
# 수식을 w로 미분한 값 : 12.0
# 수식을 w로 미분한 값 : 14.0
# 수식을 w로 미분한 값 : 16.0
# 수식을 w로 미분한 값 : 18.0
# 수식을 w로 미분한 값 : 20.0
# 수식을 w로 미분한 값 : 22.0
# 수식을 w로 미분한 값 : 24.0
# 수식을 w로 미분한 값 : 26.0
# 수식을 w로 미분한 값 : 28.0
# 수식을 w로 미분한 값 : 30.0
# 수식을 w로 미분한 값 : 32.0
# 수식을 w로 미분한 값 : 34.0
# 수식을 w로 미분한 값 : 36.0
# 수식을 w로 미분한 값 : 38.0
# 수식을 w로 미분한 값 : 40.0
# 수식을 w로 미분한 값 : 42.0

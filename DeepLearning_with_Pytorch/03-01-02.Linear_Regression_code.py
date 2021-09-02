# 전체 코드

import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim

x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[2], [4], [6]])

W = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

optimizer = optim.SGD([W, b], lr=0.001)

epochs = 2999
for e in range(epochs+1) :
    hypothesis = W * x_train + b

    cost = torch.mean((hypothesis - y_train) ** 2)

    optimizer.zero_grad()
    print(W, b)

    cost.backward()
    optimizer.step()
    # print(W, b)



    if e % 100 == 0 :
        print("Epoch {:4d}/{} W : {:3f}, b : {:.3f} Cost : {:.6f}".format(
            e, epochs, W.item(), b.item(), cost.item()
        )) 

# Epoch    0/2999 W : 0.018667, b : 0.008 Cost : 18.666666
# Epoch  100/2999 W : 1.140372, b : 0.481 Cost : 2.071139
# Epoch  200/2999 W : 1.513008, b : 0.624 Cost : 0.285313
# Epoch  300/2999 W : 1.640018, b : 0.660 Cost : 0.090525
# Epoch  400/2999 W : 1.686405, b : 0.661 Cost : 0.066789
# Epoch  500/2999 W : 1.706256, b : 0.651 Cost : 0.061560
# Epoch  600/2999 W : 1.717299, b : 0.637 Cost : 0.058446
# Epoch  700/2999 W : 1.725347, b : 0.623 Cost : 0.055677
# Epoch  800/2999 W : 1.732308, b : 0.608 Cost : 0.053061
# Epoch  900/2999 W : 1.738810, b : 0.594 Cost : 0.050570
# Epoch 1000/2999 W : 1.745061, b : 0.579 Cost : 0.048196
# Epoch 1100/2999 W : 1.751133, b : 0.566 Cost : 0.045933
# Epoch 1200/2999 W : 1.757050, b : 0.552 Cost : 0.043777
# Epoch 1300/2999 W : 1.762823, b : 0.539 Cost : 0.041721
# Epoch 1400/2999 W : 1.768458, b : 0.526 Cost : 0.039763
# Epoch 1500/2999 W : 1.773958, b : 0.514 Cost : 0.037896
# Epoch 1600/2999 W : 1.779328, b : 0.502 Cost : 0.036117
# Epoch 1700/2999 W : 1.784570, b : 0.490 Cost : 0.034421
# Epoch 1800/2999 W : 1.789688, b : 0.478 Cost : 0.032805
# Epoch 1900/2999 W : 1.794684, b : 0.467 Cost : 0.031265
# Epoch 2000/2999 W : 1.799561, b : 0.456 Cost : 0.029798
# Epoch 2100/2999 W : 1.804323, b : 0.445 Cost : 0.028399
# Epoch 2200/2999 W : 1.808971, b : 0.434 Cost : 0.027065
# Epoch 2300/2999 W : 1.813509, b : 0.424 Cost : 0.025795
# Epoch 2400/2999 W : 1.817939, b : 0.414 Cost : 0.024584
# Epoch 2500/2999 W : 1.822264, b : 0.404 Cost : 0.023430
# Epoch 2600/2999 W : 1.826486, b : 0.394 Cost : 0.022330
# Epoch 2700/2999 W : 1.830608, b : 0.385 Cost : 0.021281
# Epoch 2800/2999 W : 1.834632, b : 0.376 Cost : 0.020282
# Epoch 2900/2999 W : 1.838561, b : 0.367 Cost : 0.019330


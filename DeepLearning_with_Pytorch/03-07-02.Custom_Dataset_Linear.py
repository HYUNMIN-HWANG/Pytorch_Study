# Custom Dataset
# torch.utils.data.Dataset 을 상속받아 직접 커스텀 데이터셋을 만드는 경우

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class DataCustom(Dataset) :
    def __init__(self):
        self.x_data = torch.FloatTensor([[73, 80, 75],
                             [93, 88, 93],
                             [89, 91, 80],
                             [96, 98, 100],
                             [73, 66, 70]])
        self.y_data = torch.FloatTensor([[152], [185], [180], [196], [142]])
    
    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.x_data[idx])
        y = torch.FloatTensor(self.y_data[idx])
        return x, y

dataset = DataCustom()
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

model = nn.Linear(3, 1)
optimizer = optim.SGD(model.parameters(), lr=1e-5)

nb_epochs = 20
for epoch in range(nb_epochs+1):
    for batch_idx, samples in enumerate(dataloader):
        x_train, y_train = samples

        prediction = model(x_train)

        cost = F.mse_loss(prediction, y_train)

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        print("Epoch {:4d}/{} | Batch {}/{} | Cost {:.6f}".format(
            epoch, nb_epochs, batch_idx, len(dataloader), cost.item()
        ))

# Epoch    0/20 | Batch 0/3 | Cost 29556.441406
# Epoch    0/20 | Batch 1/3 | Cost 10028.322266
# Epoch    0/20 | Batch 2/3 | Cost 5131.451172
# Epoch    1/20 | Batch 0/3 | Cost 892.401001
# Epoch    1/20 | Batch 1/3 | Cost 203.510895
# Epoch    1/20 | Batch 2/3 | Cost 88.309402
# Epoch    2/20 | Batch 0/3 | Cost 23.797411
# Epoch    2/20 | Batch 1/3 | Cost 23.124958
# Epoch    2/20 | Batch 2/3 | Cost 24.247540
# Epoch    3/20 | Batch 0/3 | Cost 8.593388
# Epoch    3/20 | Batch 1/3 | Cost 10.556912
# Epoch    3/20 | Batch 2/3 | Cost 8.587629
# Epoch    4/20 | Batch 0/3 | Cost 13.221855
# Epoch    4/20 | Batch 1/3 | Cost 5.361130
# Epoch    4/20 | Batch 2/3 | Cost 12.201876
# Epoch    5/20 | Batch 0/3 | Cost 9.743957
# Epoch    5/20 | Batch 1/3 | Cost 12.773077
# Epoch    5/20 | Batch 2/3 | Cost 6.123843
# Epoch    6/20 | Batch 0/3 | Cost 9.410993
# Epoch    6/20 | Batch 1/3 | Cost 13.565358
# Epoch    6/20 | Batch 2/3 | Cost 1.644731
# Epoch    7/20 | Batch 0/3 | Cost 1.641299
# Epoch    7/20 | Batch 1/3 | Cost 9.679059
# Epoch    7/20 | Batch 2/3 | Cost 25.014803
# Epoch    8/20 | Batch 0/3 | Cost 8.712861
# Epoch    8/20 | Batch 1/3 | Cost 10.437280
# Epoch    8/20 | Batch 2/3 | Cost 9.490547
# Epoch    9/20 | Batch 0/3 | Cost 12.831049
# Epoch    9/20 | Batch 1/3 | Cost 5.329350
# Epoch    9/20 | Batch 2/3 | Cost 12.221604
# Epoch   10/20 | Batch 0/3 | Cost 3.646839
# Epoch   10/20 | Batch 1/3 | Cost 18.308161
# Epoch   10/20 | Batch 2/3 | Cost 8.623260
# Epoch   11/20 | Batch 0/3 | Cost 9.457167
# Epoch   11/20 | Batch 1/3 | Cost 9.424437
# Epoch   11/20 | Batch 2/3 | Cost 10.385219
# Epoch   12/20 | Batch 0/3 | Cost 8.800123
# Epoch   12/20 | Batch 1/3 | Cost 11.617649
# Epoch   12/20 | Batch 2/3 | Cost 6.998712
# Epoch   13/20 | Batch 0/3 | Cost 14.356057
# Epoch   13/20 | Batch 1/3 | Cost 5.819791
# Epoch   13/20 | Batch 2/3 | Cost 12.211258
# Epoch   14/20 | Batch 0/3 | Cost 9.074351
# Epoch   14/20 | Batch 1/3 | Cost 9.543720
# Epoch   14/20 | Batch 2/3 | Cost 8.226015
# Epoch   15/20 | Batch 0/3 | Cost 6.251847
# Epoch   15/20 | Batch 1/3 | Cost 11.375319
# Epoch   15/20 | Batch 2/3 | Cost 13.615442
# Epoch   16/20 | Batch 0/3 | Cost 5.853995
# Epoch   16/20 | Batch 1/3 | Cost 12.467746
# Epoch   16/20 | Batch 2/3 | Cost 12.707067
# Epoch   17/20 | Batch 0/3 | Cost 3.216663
# Epoch   17/20 | Batch 1/3 | Cost 8.601280
# Epoch   17/20 | Batch 2/3 | Cost 24.714565
# Epoch   18/20 | Batch 0/3 | Cost 8.636171
# Epoch   18/20 | Batch 1/3 | Cost 7.128094
# Epoch   18/20 | Batch 2/3 | Cost 15.328009
# Epoch   19/20 | Batch 0/3 | Cost 5.157955
# Epoch   19/20 | Batch 1/3 | Cost 15.326164
# Epoch   19/20 | Batch 2/3 | Cost 6.191240
# Epoch   20/20 | Batch 0/3 | Cost 19.491398
# Epoch   20/20 | Batch 1/3 | Cost 7.776244
# Epoch   20/20 | Batch 2/3 | Cost 9.547512
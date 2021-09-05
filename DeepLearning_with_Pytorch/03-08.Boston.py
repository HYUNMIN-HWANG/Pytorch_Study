# Boston 집 값 예측

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd

from torch.utils.data import TensorDataset, dataloader
from torch.utils.data import DataLoader

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

'''DATA'''
boston = load_boston()
print(boston.keys())
# dict_keys(['data', 'target', 'feature_names', 'DESCR', 'filename'])

boston_df = pd.DataFrame(boston['data'])
print(boston_df.head())
#         0     1     2    3      4      5     6       7    8      9     10      11    12
# 0  0.00632  18.0  2.31  0.0  0.538  6.575  65.2  4.0900  1.0  296.0  15.3  396.90  4.98
# 1  0.02731   0.0  7.07  0.0  0.469  6.421  78.9  4.9671  2.0  242.0  17.8  396.90  9.14
# 2  0.02729   0.0  7.07  0.0  0.469  7.185  61.1  4.9671  2.0  242.0  17.8  392.83  4.03
# 3  0.03237   0.0  2.18  0.0  0.458  6.998  45.8  6.0622  3.0  222.0  18.7  394.63  2.94
# 4  0.06905   0.0  2.18  0.0  0.458  7.147  54.2  6.0622  3.0  222.0  18.7  396.90  5.33

boston_df.columns = pd.DataFrame(boston['feature_names'])
print(boston_df.head())
#    (CRIM,)  (ZN,)  (INDUS,)  (CHAS,)  (NOX,)  (RM,)  (AGE,)  (DIS,)  (RAD,)  (TAX,)  (PTRATIO,)    (B,)  (LSTAT,)
# 0  0.00632   18.0      2.31      0.0   0.538  6.575    65.2  4.0900     1.0   296.0        15.3  396.90      4.98
# 1  0.02731    0.0      7.07      0.0   0.469  6.421    78.9  4.9671     2.0   242.0        17.8  396.90      9.14
# 2  0.02729    0.0      7.07      0.0   0.469  7.185    61.1  4.9671     2.0   242.0        17.8  392.83      4.03
# 3  0.03237    0.0      2.18      0.0   0.458  6.998    45.8  6.0622     3.0   222.0        18.7  394.63      2.94
# 4  0.06905    0.0      2.18      0.0   0.458  7.147    54.2  6.0622     3.0   222.0        18.7  396.90      5.33

boston_df['PRICE'] = boston['target']
print(boston_df.head())
#    (CRIM,)  (ZN,)  (INDUS,)  (CHAS,)  (NOX,)  (RM,)  (AGE,)  (DIS,)  (RAD,)  (TAX,)  (PTRATIO,)    (B,)  (LSTAT,)  PRICE
# 0  0.00632   18.0      2.31      0.0   0.538  6.575    65.2  4.0900     1.0   296.0        15.3  396.90      4.98   24.0
# 1  0.02731    0.0      7.07      0.0   0.469  6.421    78.9  4.9671     2.0   242.0        17.8  396.90      9.14   21.6
# 2  0.02729    0.0      7.07      0.0   0.469  7.185    61.1  4.9671     2.0   242.0        17.8  392.83      4.03   34.7
# 3  0.03237    0.0      2.18      0.0   0.458  6.998    45.8  6.0622     3.0   222.0        18.7  394.63      2.94   33.4
# 4  0.06905    0.0      2.18      0.0   0.458  7.147    54.2  6.0622     3.0   222.0        18.7  396.90      5.33   36.2

x = boston_df.iloc[:,:-1]
y = boston_df['PRICE']

x = x.to_numpy()
y = y.to_numpy()

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2021, shuffle=True)
print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)
# (404, 13) (102, 13)
# (404,) (102,)

n_train = X_train.shape[0]
X_train = torch.tensor(X_train, dtype=torch.float)
X_test = torch.tensor(X_test, dtype=torch.float)
Y_train = torch.tensor(y_train, dtype=torch.float).view(-1, 1)
Y_test = torch.tensor(y_test, dtype=torch.float).view(-1, 1)

batch_size = 8

datasets = TensorDataset(X_train, Y_train)
train_iter = DataLoader(datasets, batch_size=batch_size, shuffle=True)

'''Model'''
nb_epoch = 400
learnin_rate = 0.001
size_hidden = 32

batch_no = len(X_train) // batch_size
cols = X_train.shape[1]
n_output = 1

class Net(nn.Module): 
    def __init__(self, cols, size_hidden, n_output):
        super().__init__()              
        self.fc1 = nn.Linear(cols, size_hidden)
        self.fc2 = nn.Linear(size_hidden, size_hidden*2)
        self.fc3 = nn.Linear(size_hidden*2, size_hidden)
        self.fc4 = nn.Linear(size_hidden, 16)
        self.predict = nn.Linear(16, n_output)

    def forward(self, x):        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.predict(x)       
        return x

model = Net(cols, size_hidden, n_output)
print(model)
# Net(
#   (fc1): Linear(in_features=13, out_features=32, bias=True)
#   (fc2): Linear(in_features=32, out_features=64, bias=True)
#   (fc3): Linear(in_features=64, out_features=32, bias=True)
#   (fc4): Linear(in_features=32, out_features=16, bias=True)
#   (predict): Linear(in_features=16, out_features=1, bias=True)
# )


optimizer = optim.Adam(model.parameters(), lr=learnin_rate, weight_decay=1e-7)
loss = torch.nn.MSELoss()
n = len(train_iter)

for epoch in range(nb_epoch+1):
    running_loss = 0.0
    for x, y in train_iter :
        prediction = model(x)
        # cost = loss(prediction, y)
        cost = F.mse_loss(prediction, y)

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        running_loss += cost.item()

    print("Epoch {:4d}/{} | Cost {:.6f}".format(
        epoch, nb_epoch, running_loss/n
    ))


# Epoch    0/400 | Cost 180.525434
# Epoch    1/400 | Cost 71.956207
# Epoch    2/400 | Cost 68.592476
# Epoch    3/400 | Cost 64.788221
# Epoch    4/400 | Cost 63.255243
# Epoch    5/400 | Cost 60.623503
# Epoch    6/400 | Cost 62.474809
# Epoch    7/400 | Cost 56.895212
# Epoch    8/400 | Cost 58.416164
# Epoch    9/400 | Cost 55.132971
# Epoch   10/400 | Cost 55.205195
# Epoch   11/400 | Cost 56.523559
# Epoch   12/400 | Cost 48.928252
# Epoch   13/400 | Cost 45.983364
# Epoch   14/400 | Cost 49.268592
# Epoch   15/400 | Cost 47.036259
# Epoch   16/400 | Cost 41.109126
# Epoch   17/400 | Cost 41.473266
# Epoch   18/400 | Cost 35.865898
# Epoch   19/400 | Cost 40.711999
# Epoch   20/400 | Cost 35.987887
# Epoch   21/400 | Cost 32.233297
# Epoch   22/400 | Cost 29.685835
# Epoch   23/400 | Cost 29.268890
# Epoch   24/400 | Cost 29.050794
# Epoch   25/400 | Cost 27.304772
# Epoch   26/400 | Cost 29.698473
# Epoch   27/400 | Cost 27.555784
# Epoch   28/400 | Cost 27.396323
# Epoch   29/400 | Cost 28.238267
# Epoch   30/400 | Cost 24.346330
# Epoch   31/400 | Cost 22.414781
# Epoch   32/400 | Cost 23.203932
# Epoch   33/400 | Cost 22.571577
# Epoch   34/400 | Cost 21.769681
# Epoch   35/400 | Cost 21.452430
# Epoch   36/400 | Cost 22.057072
# Epoch   37/400 | Cost 22.368493
# Epoch   38/400 | Cost 20.174592
# Epoch   39/400 | Cost 22.936838
# Epoch   40/400 | Cost 20.631103
# Epoch   41/400 | Cost 23.757711
# Epoch   42/400 | Cost 20.188508
# Epoch   43/400 | Cost 20.778943
# Epoch   44/400 | Cost 19.723531
# Epoch   45/400 | Cost 17.843031
# Epoch   46/400 | Cost 22.113292
# Epoch   47/400 | Cost 21.645059
# Epoch   48/400 | Cost 23.380270
# Epoch   49/400 | Cost 20.711138
# Epoch   50/400 | Cost 17.527906
# Epoch   51/400 | Cost 18.638314
# Epoch   52/400 | Cost 17.974187
# Epoch   53/400 | Cost 19.057580
# Epoch   54/400 | Cost 17.419384
# Epoch   55/400 | Cost 17.486502
# Epoch   56/400 | Cost 18.308895
# Epoch   57/400 | Cost 17.462474
# Epoch   58/400 | Cost 17.190130
# Epoch   59/400 | Cost 18.888601
# Epoch   60/400 | Cost 15.732258
# Epoch   61/400 | Cost 19.219823
# Epoch   62/400 | Cost 18.019417
# Epoch   63/400 | Cost 20.472112
# Epoch   64/400 | Cost 25.987185
# Epoch   65/400 | Cost 19.445260
# Epoch   66/400 | Cost 21.279398
# Epoch   67/400 | Cost 19.680137
# Epoch   68/400 | Cost 15.899561
# Epoch   69/400 | Cost 15.622298
# Epoch   70/400 | Cost 18.578007
# Epoch   71/400 | Cost 15.552148
# Epoch   72/400 | Cost 14.948178
# Epoch   73/400 | Cost 18.007866
# Epoch   74/400 | Cost 15.807330
# Epoch   75/400 | Cost 15.967787
# Epoch   76/400 | Cost 17.239177
# Epoch   77/400 | Cost 16.807913
# Epoch   78/400 | Cost 16.477745
# Epoch   79/400 | Cost 15.020080
# Epoch   80/400 | Cost 17.030193
# Epoch   81/400 | Cost 15.861438
# Epoch   82/400 | Cost 17.932265
# Epoch   83/400 | Cost 14.919826
# Epoch   84/400 | Cost 16.058349
# Epoch   85/400 | Cost 13.987519
# Epoch   86/400 | Cost 15.796765
# Epoch   87/400 | Cost 16.580598
# Epoch   88/400 | Cost 21.189650
# Epoch   89/400 | Cost 15.071832
# Epoch   90/400 | Cost 19.254621
# Epoch   91/400 | Cost 17.371549
# Epoch   92/400 | Cost 14.842534
# Epoch   93/400 | Cost 14.943596
# Epoch   94/400 | Cost 16.033239
# Epoch   95/400 | Cost 15.575436
# Epoch   96/400 | Cost 13.722372
# Epoch   97/400 | Cost 14.477956
# Epoch   98/400 | Cost 14.700573
# Epoch   99/400 | Cost 15.484101
# Epoch  100/400 | Cost 19.803733
# Epoch  101/400 | Cost 13.993691
# Epoch  102/400 | Cost 16.498639
# Epoch  103/400 | Cost 14.579032
# Epoch  104/400 | Cost 13.967849
# Epoch  105/400 | Cost 14.821226
# Epoch  106/400 | Cost 14.988572
# Epoch  107/400 | Cost 12.915228
# Epoch  108/400 | Cost 13.817956
# Epoch  109/400 | Cost 14.185931
# Epoch  110/400 | Cost 15.382577
# Epoch  111/400 | Cost 12.482565
# Epoch  112/400 | Cost 17.537172
# Epoch  113/400 | Cost 14.564062
# Epoch  114/400 | Cost 13.409471
# Epoch  115/400 | Cost 13.905308
# Epoch  116/400 | Cost 13.620678
# Epoch  117/400 | Cost 12.382852
# Epoch  118/400 | Cost 12.995549
# Epoch  119/400 | Cost 14.425694
# Epoch  120/400 | Cost 13.178496
# Epoch  121/400 | Cost 17.028873
# Epoch  122/400 | Cost 14.554730
# Epoch  123/400 | Cost 13.105320
# Epoch  124/400 | Cost 15.535841
# Epoch  125/400 | Cost 12.932346
# Epoch  126/400 | Cost 12.188485
# Epoch  127/400 | Cost 15.293918
# Epoch  128/400 | Cost 12.854385
# Epoch  129/400 | Cost 12.383524
# Epoch  130/400 | Cost 14.554904
# Epoch  131/400 | Cost 12.153690
# Epoch  132/400 | Cost 12.983403
# Epoch  133/400 | Cost 17.246173
# Epoch  134/400 | Cost 16.468014
# Epoch  135/400 | Cost 11.510055
# Epoch  136/400 | Cost 14.157072
# Epoch  137/400 | Cost 11.178475
# Epoch  138/400 | Cost 12.030191
# Epoch  139/400 | Cost 13.157137
# Epoch  140/400 | Cost 12.634446
# Epoch  141/400 | Cost 12.098976
# Epoch  142/400 | Cost 13.269541
# Epoch  143/400 | Cost 12.276521
# Epoch  144/400 | Cost 12.356574
# Epoch  145/400 | Cost 11.951444
# Epoch  146/400 | Cost 11.371044
# Epoch  147/400 | Cost 12.247533
# Epoch  148/400 | Cost 15.049913
# Epoch  149/400 | Cost 15.448648
# Epoch  150/400 | Cost 12.414569
# Epoch  151/400 | Cost 11.940946
# Epoch  152/400 | Cost 13.529347
# Epoch  153/400 | Cost 12.942195
# Epoch  154/400 | Cost 22.620773
# Epoch  155/400 | Cost 11.224700
# Epoch  156/400 | Cost 11.812080
# Epoch  157/400 | Cost 11.615228
# Epoch  158/400 | Cost 15.593862
# Epoch  159/400 | Cost 15.314209
# Epoch  160/400 | Cost 13.525422
# Epoch  161/400 | Cost 12.440284
# Epoch  162/400 | Cost 15.786564
# Epoch  163/400 | Cost 14.223841
# Epoch  164/400 | Cost 14.537356
# Epoch  165/400 | Cost 13.157556
# Epoch  166/400 | Cost 13.593065
# Epoch  167/400 | Cost 11.607105
# Epoch  168/400 | Cost 10.044169
# Epoch  169/400 | Cost 12.571691
# Epoch  170/400 | Cost 13.521908
# Epoch  171/400 | Cost 11.003828
# Epoch  172/400 | Cost 10.934163
# Epoch  173/400 | Cost 10.865188
# Epoch  174/400 | Cost 10.370403
# Epoch  175/400 | Cost 11.437398
# Epoch  176/400 | Cost 10.919575
# Epoch  177/400 | Cost 10.838407
# Epoch  178/400 | Cost 13.433614
# Epoch  179/400 | Cost 10.924324
# Epoch  180/400 | Cost 9.853949
# Epoch  181/400 | Cost 11.072368
# Epoch  182/400 | Cost 11.350613
# Epoch  183/400 | Cost 11.337215
# Epoch  184/400 | Cost 10.580514
# Epoch  185/400 | Cost 13.445862
# Epoch  186/400 | Cost 11.316587
# Epoch  187/400 | Cost 11.873276
# Epoch  188/400 | Cost 13.911949
# Epoch  189/400 | Cost 10.320544
# Epoch  190/400 | Cost 12.055960
# Epoch  191/400 | Cost 13.171907
# Epoch  192/400 | Cost 12.836229
# Epoch  193/400 | Cost 10.920437
# Epoch  194/400 | Cost 10.268167
# Epoch  195/400 | Cost 10.099056
# Epoch  196/400 | Cost 10.468052
# Epoch  197/400 | Cost 10.683204
# Epoch  198/400 | Cost 11.344137
# Epoch  199/400 | Cost 12.579203
# Epoch  200/400 | Cost 14.788983
# Epoch  201/400 | Cost 12.986376
# Epoch  202/400 | Cost 13.064836
# Epoch  203/400 | Cost 12.327487
# Epoch  204/400 | Cost 10.844261
# Epoch  205/400 | Cost 11.895766
# Epoch  206/400 | Cost 11.287517
# Epoch  207/400 | Cost 9.576661
# Epoch  208/400 | Cost 10.048683
# Epoch  209/400 | Cost 13.201292
# Epoch  210/400 | Cost 12.529701
# Epoch  211/400 | Cost 10.878787
# Epoch  212/400 | Cost 11.598203
# Epoch  213/400 | Cost 11.789001
# Epoch  214/400 | Cost 12.962957
# Epoch  215/400 | Cost 10.109916
# Epoch  216/400 | Cost 9.441751
# Epoch  217/400 | Cost 11.226414
# Epoch  218/400 | Cost 12.686219
# Epoch  219/400 | Cost 10.632139
# Epoch  220/400 | Cost 10.343168
# Epoch  221/400 | Cost 9.332622
# Epoch  222/400 | Cost 11.381802
# Epoch  223/400 | Cost 9.438835
# Epoch  224/400 | Cost 11.804203
# Epoch  225/400 | Cost 10.991194
# Epoch  226/400 | Cost 9.882075
# Epoch  227/400 | Cost 9.622430
# Epoch  228/400 | Cost 10.461930
# Epoch  229/400 | Cost 10.038897
# Epoch  230/400 | Cost 11.172263
# Epoch  231/400 | Cost 9.299934
# Epoch  232/400 | Cost 9.235250
# Epoch  233/400 | Cost 14.913732
# Epoch  234/400 | Cost 11.642434
# Epoch  235/400 | Cost 13.350963
# Epoch  236/400 | Cost 9.394297
# Epoch  237/400 | Cost 19.013790
# Epoch  238/400 | Cost 10.513302
# Epoch  239/400 | Cost 9.884037
# Epoch  240/400 | Cost 10.114781
# Epoch  241/400 | Cost 9.078704
# Epoch  242/400 | Cost 11.742546
# Epoch  243/400 | Cost 15.076550
# Epoch  244/400 | Cost 9.905910
# Epoch  245/400 | Cost 9.209322
# Epoch  246/400 | Cost 8.866594
# Epoch  247/400 | Cost 9.936407
# Epoch  248/400 | Cost 9.185148
# Epoch  249/400 | Cost 9.463335
# Epoch  250/400 | Cost 9.370410
# Epoch  251/400 | Cost 10.470061
# Epoch  252/400 | Cost 9.001040
# Epoch  253/400 | Cost 19.056740
# Epoch  254/400 | Cost 11.738911
# Epoch  255/400 | Cost 10.552407
# Epoch  256/400 | Cost 10.505115
# Epoch  257/400 | Cost 8.642683
# Epoch  258/400 | Cost 10.187509
# Epoch  259/400 | Cost 10.952834
# Epoch  260/400 | Cost 9.670863
# Epoch  261/400 | Cost 8.432428
# Epoch  262/400 | Cost 9.428979
# Epoch  263/400 | Cost 10.488671
# Epoch  264/400 | Cost 10.882768
# Epoch  265/400 | Cost 10.822968
# Epoch  266/400 | Cost 8.885853
# Epoch  267/400 | Cost 10.378069
# Epoch  268/400 | Cost 10.477278
# Epoch  269/400 | Cost 10.442324
# Epoch  270/400 | Cost 9.724706
# Epoch  271/400 | Cost 11.292088
# Epoch  272/400 | Cost 9.758847
# Epoch  273/400 | Cost 10.655895
# Epoch  274/400 | Cost 8.220543
# Epoch  275/400 | Cost 8.515673
# Epoch  276/400 | Cost 7.860818
# Epoch  277/400 | Cost 7.176781
# Epoch  278/400 | Cost 13.789402
# Epoch  279/400 | Cost 11.540828
# Epoch  280/400 | Cost 7.489251
# Epoch  281/400 | Cost 8.838210
# Epoch  282/400 | Cost 7.531103
# Epoch  283/400 | Cost 9.594244
# Epoch  284/400 | Cost 13.038731
# Epoch  285/400 | Cost 8.801145
# Epoch  286/400 | Cost 8.650310
# Epoch  287/400 | Cost 7.961252
# Epoch  288/400 | Cost 10.818103
# Epoch  289/400 | Cost 12.226283
# Epoch  290/400 | Cost 8.604300
# Epoch  291/400 | Cost 10.142860
# Epoch  292/400 | Cost 9.403785
# Epoch  293/400 | Cost 7.834922
# Epoch  294/400 | Cost 9.092138
# Epoch  295/400 | Cost 9.473504
# Epoch  296/400 | Cost 8.120109
# Epoch  297/400 | Cost 8.240959
# Epoch  298/400 | Cost 10.031671
# Epoch  299/400 | Cost 7.399174
# Epoch  300/400 | Cost 8.461817
# Epoch  301/400 | Cost 11.498868
# Epoch  302/400 | Cost 11.939274
# Epoch  303/400 | Cost 13.005965
# Epoch  304/400 | Cost 10.940342
# Epoch  305/400 | Cost 8.234588
# Epoch  306/400 | Cost 7.344773
# Epoch  307/400 | Cost 7.611583
# Epoch  308/400 | Cost 10.049044
# Epoch  309/400 | Cost 11.907863
# Epoch  310/400 | Cost 11.596718
# Epoch  311/400 | Cost 7.115157
# Epoch  312/400 | Cost 6.878322
# Epoch  313/400 | Cost 7.626829
# Epoch  314/400 | Cost 7.358459
# Epoch  315/400 | Cost 8.913684
# Epoch  316/400 | Cost 7.991464
# Epoch  317/400 | Cost 10.971889
# Epoch  318/400 | Cost 8.098524
# Epoch  319/400 | Cost 9.276057
# Epoch  320/400 | Cost 8.571970
# Epoch  321/400 | Cost 7.927551
# Epoch  322/400 | Cost 8.020935
# Epoch  323/400 | Cost 7.662294
# Epoch  324/400 | Cost 6.965587
# Epoch  325/400 | Cost 8.100167
# Epoch  326/400 | Cost 8.794980
# Epoch  327/400 | Cost 9.140243
# Epoch  328/400 | Cost 7.579449
# Epoch  329/400 | Cost 8.308515
# Epoch  330/400 | Cost 9.608501
# Epoch  331/400 | Cost 9.689533
# Epoch  332/400 | Cost 8.557859
# Epoch  333/400 | Cost 16.370662
# Epoch  334/400 | Cost 8.347798
# Epoch  335/400 | Cost 10.710302
# Epoch  336/400 | Cost 10.387544
# Epoch  337/400 | Cost 10.490919
# Epoch  338/400 | Cost 9.731587
# Epoch  339/400 | Cost 7.565262
# Epoch  340/400 | Cost 9.259611
# Epoch  341/400 | Cost 6.670019
# Epoch  342/400 | Cost 8.084787
# Epoch  343/400 | Cost 10.003900
# Epoch  344/400 | Cost 11.472398
# Epoch  345/400 | Cost 7.190259
# Epoch  346/400 | Cost 9.172503
# Epoch  347/400 | Cost 11.196053
# Epoch  348/400 | Cost 9.411177
# Epoch  349/400 | Cost 7.026098
# Epoch  350/400 | Cost 7.372174
# Epoch  351/400 | Cost 9.007016
# Epoch  352/400 | Cost 7.064937
# Epoch  353/400 | Cost 6.992470
# Epoch  354/400 | Cost 6.691614
# Epoch  355/400 | Cost 6.367919
# Epoch  356/400 | Cost 7.261381
# Epoch  357/400 | Cost 11.096625
# Epoch  358/400 | Cost 8.391243
# Epoch  359/400 | Cost 10.878730
# Epoch  360/400 | Cost 8.868954
# Epoch  361/400 | Cost 11.907495
# Epoch  362/400 | Cost 7.988925
# Epoch  363/400 | Cost 9.057302
# Epoch  364/400 | Cost 6.842564
# Epoch  365/400 | Cost 8.435225
# Epoch  366/400 | Cost 7.218088
# Epoch  367/400 | Cost 8.664294
# Epoch  368/400 | Cost 13.244361
# Epoch  369/400 | Cost 12.027074
# Epoch  370/400 | Cost 10.773255
# Epoch  371/400 | Cost 8.076167
# Epoch  372/400 | Cost 10.757730
# Epoch  373/400 | Cost 9.928824
# Epoch  374/400 | Cost 8.839805
# Epoch  375/400 | Cost 9.834281
# Epoch  376/400 | Cost 9.477004
# Epoch  377/400 | Cost 7.555092
# Epoch  378/400 | Cost 6.771330
# Epoch  379/400 | Cost 7.140688
# Epoch  380/400 | Cost 6.591829
# Epoch  381/400 | Cost 7.443061
# Epoch  382/400 | Cost 9.121924
# Epoch  383/400 | Cost 7.707494
# Epoch  384/400 | Cost 9.086377
# Epoch  385/400 | Cost 6.390821
# Epoch  386/400 | Cost 6.078678
# Epoch  387/400 | Cost 7.975766
# Epoch  388/400 | Cost 13.921802
# Epoch  389/400 | Cost 7.589183
# Epoch  390/400 | Cost 7.909663
# Epoch  391/400 | Cost 7.808401
# Epoch  392/400 | Cost 7.154664
# Epoch  393/400 | Cost 7.174931
# Epoch  394/400 | Cost 10.048483
# Epoch  395/400 | Cost 7.283839
# Epoch  396/400 | Cost 6.337205
# Epoch  397/400 | Cost 6.143502
# Epoch  398/400 | Cost 6.485214
# Epoch  399/400 | Cost 7.238074
# Epoch  400/400 | Cost 5.979768
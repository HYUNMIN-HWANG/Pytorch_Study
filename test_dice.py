import torch 
import torch.nn as nn
import ipdb

def round_fn(pred) :
    pred = pred.view(-1).float()
    zero_flat = torch.zeros(262144).to(torch.device('cuda'))   
    one_flat = torch.ones(262144).to(torch.device('cuda'))   
    pred = torch.where(pred>=0.5, one_flat, pred)   # 0.5 이상인 것 1.0으로 바꾸기 
    pred = torch.where(pred<0.5, zero_flat, pred)
    # pred = torch.round(pred)
    return pred


def loss_fn(pred, target, criterion, smooth = 1e-5):

    # ipdb.set_trace()

    # binary cross entropy loss
    bce = criterion(pred, target)

    # dice coefficient
    pred = pred.view(-1).float()
    #pred = pred.to(torch.device('cpu'))  
    #pred = pred.detach().numpy()

    target = target.view(-1).float()
    #target = target.to(torch.device('cpu'))  
    #target = target.detach().numpy()

    # union = (target + pred).sum(-1)
    # intersection = (target * pred).sum(-1)
    # dice = (2. * intersection+smooth) / (union + smooth)
    # dice_loss = 1 - dice    # 원래 mean 했었는데 mean 안 해도 되나??

    # intersection = torch.zeros(len(pred)).to(torch.device('cuda'))  
    # union = torch.zeros(len(pred)).to(torch.device('cuda'))  

    # for i in range(len(pred)) :
    #      # intersection
    #     if target[i] == 1 and pred[i] == 1 :
    #         intersection[i] = 1
        
    #     # union
    #     if target[i] == 1 or pred[i] == 1:
    #         union[i] = 1

    # one_vector = torch.ones(len(pred)).to(torch.device('cuda'))  
    # zero_vector = torch.zeros(len(pred)).to(torch.device('cuda'))  

    # union = (target + pred).to(torch.device('cuda'))  
    # intersection = torch.where(union == 2.0, one_vector, zero_vector)
    # union = torch.where(sum_matrix == 2.0, one_vector, sum_matrix)

    # print(intersection)
    # intersection = intersection.float().sum()
    # print(union)
    # union = union.float().sum()
    # dice = (2. * intersection + smooth) / (union+ smooth) 


    intersection = 2.0 * (target * pred).sum()
    union = target.sum() + pred.sum()
    if target.sum() == 0 and pred.sum() == 0:
        return bce.sum()+1.0, bce.sum(), 1.0
    dice = intersection / union

    dice_loss = 1 - dice    # 원래 mean 했었는데 mean 안 해도 되나??
    total_cost = bce.sum() + dice_loss.sum()

    return  total_cost, bce.sum(), dice.sum()


pred = torch.Tensor([[1,0],[0,0],[0,1]])
target = torch.Tensor([[1,0],[0,0],[0,1]])
total, bce, diceloss = loss_fn(pred, target, nn.BCELoss())
print("total ", total, " | bce ", bce, " | Dice ", diceloss)

pred = torch.Tensor([[1,0],[0,0],[0,1]])
target = torch.Tensor([[0,1],[1,1],[1,0]])
total, bce, diceloss = loss_fn(pred, target, nn.BCELoss())
print("total ", total, " | bce ", bce, " | Dice", diceloss)

pred = torch.Tensor([[1,0],[0,0],[0,1]])
target = torch.Tensor([[1,1],[1,1],[1,1]])
total, bce, diceloss = loss_fn(pred, target, nn.BCELoss())
print("total ", total, " | bce ", bce, " | Dice ", diceloss)

pred = torch.Tensor([[0,0],[0,0],[0,0]])
target = torch.Tensor([[0,0],[0,0],[0,0]])
total, bce, diceloss = loss_fn(pred, target, nn.BCELoss())
print("total ", total, " | bce ", bce, " | Dice ", diceloss)


pred = torch.Tensor([1,1,0,0,1,1,1,0])
target = torch.Tensor([0,1,1,0,0,1,1,0])
total, bce, diceloss = loss_fn(pred, target, nn.BCELoss())
print("total ", total, " | bce ", bce, " | Dice ", diceloss)
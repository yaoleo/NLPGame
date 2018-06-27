import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset

loss_fn = torch.nn.BCELoss(reduce=False, size_average=False)
input = Variable(torch.randn(3, 4))
target = Variable(torch.FloatTensor(3, 4).random_(2))
loss = loss_fn(F.sigmoid(input), target)
print(input); print(target); print(loss)

class_weight = Variable(torch.FloatTensor([1, 10]))
target = Variable(torch.FloatTensor(3, 4).random_(2))
weight = class_weight[target.long()] # (3, 4)
loss_fn = torch.nn.BCELoss(weight=weight, reduce=False, size_average=False)
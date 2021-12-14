# Custom Sequential

import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import pandas as pd

def freeze_parameters(model):
    for param in model.parameters():
        param.requires_grad = False
    return model

def unfreeze_parameters(model):
    for param in model.parameters():
        param.requires_grad = True
    return model

class explain_model(nn.Module):
    def __init__(self, module_list):
        super().__init__()
        self.module_list = module_list
        self.num_modules = len(self.module_list)
        for cur_index in range(self.num_modules):
            self.module_list[cur_index] = freeze_parameters(self.module_list[cur_index])
        self.freeze_index = dict((x, True) for x in range(self.num_modules))
        
    def forward(self, x, active_index):
        if self.freeze_index[active_index]==True:
            self.module_list[active_index] = unfreeze_parameters(self.module_list[active_index])
            if active_index-1 >= 0:
                self.module_list[active_index-1] = freeze_parameters(self.module_list[active_index-1])
        output_flag = False
        for cur_index in range(active_index+1):
            if output_flag is False:
                output = self.module_list[cur_index](X)
                output_flag = True
            else:
                output += self.module_list[cur_index](X)
        return output
            

class model1(nn.Module)        :
    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features
        self.seq = nn.Sequential(
            nn.Linear(self.num_features, 1)
        )
        
    def forward(self, x):
        return self.seq(x)
        
class model2(nn.Module)        :
    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features
        self.seq = nn.Sequential(
            nn.Linear(self.num_features, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(num_features, 1)
        )
        
    def forward(self, x):
        return self.seq(x)
        
# Init
num_features=100
num_modules = 2
model1 = model1(num_features)
model2 = model2(num_features)
model = explain_model(nn.ModuleList([model1, model2]))
criterion = nn.MSELoss()

# Random Data
X = torch.rand(100, 100)
y = torch.rand(100,).type(torch.FloatTensor)

# Num Epochs
num_epochs = 1000
learning_rate = 1e-2
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

def increment_index(cur_loss, prev_loss):
    if ((prev_loss - cur_loss) < 0.0001) and (cur_loss < prev_loss):
        return True

active_index = 0
prev_loss = 9999

for cur_epoch in range(num_epochs):
    model.zero_grad()
    output = model(X, active_index)
    loss = criterion(y, output)
    if increment_index(loss.item(), prev_loss) == True:
        if active_index < num_modules-1: 
            active_index = active_index + 1
    loss.backward()
    optimizer.step()
    prev_loss = loss.item()
    print("Cur Epoch {0} loss is {1} and active index {2} loss_diff {3}".format(cur_epoch, loss.item(), active_index, prev_loss - loss.item()))

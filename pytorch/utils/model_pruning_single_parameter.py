import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.utils.prune as prune

class random_model(nn.Module):
    def __init__(self, num_layers):
        super(random_model, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(100, 20), nn.BatchNorm1d(20), nn.ReLU())
        self.layer2 = nn.Linear(20, 1)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x
    
model = random_model(10)
loss = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


X = torch.rand(100, 100)
y = torch.rand(100)

for cur_epoch in range(100):
    optimizer.zero_grad()
    output = model(X)
    cur_loss = loss(output, y)
    cur_loss.backward()
    optimizer.step()
print("Epoch {0} Loss is {1}".format(cur_epoch, cur_loss.item()))

# After Training
output = model(X)
cur_loss = loss(output, y)
print("Loss is {0}".format(cur_loss.item()))

# Module
module = model._modules['layer2']
model._modules['layer2'] = prune.random_unstructured(module, name="weight", amount=0.3)
output = model(X)
cur_loss = loss(output, y)
print("Loss is {0}".format(cur_loss.item()))
print(list(module.named_buffers()))

# Resetting, so that the mask is gone
module = model._modules['layer2']
prune.remove(module, 'weight')
print(list(module.named_buffers()))
output = model(X)
cur_loss = loss(output, y)
print("Loss is {0}".format(cur_loss.item()))

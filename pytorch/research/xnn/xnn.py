#!wget https://github.com/mctorch/mctorch/archive/refs/heads/master.zip
#!pip install https://github.com/mctorch/mctorch/archive/refs/heads/master.zip
#!pip install torch

import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
import torch

from sklearn import cluster, datasets, mixture
blobs = datasets.make_blobs(n_samples=100, n_features=100, random_state=8, centers=8)
X = torch.from_numpy(blobs[0]).type(torch.FloatTensor)
y = torch.from_numpy(blobs[1]).type(torch.FloatTensor)

# EXPLANABLE NEURAL NETWORK
import os
import numpy as np
import time
import torch
import torchvision
from torch import nn
import mctorch.nn as mnn
from torch.autograd import Variable
import torch.nn.functional as F

torch.manual_seed(42)

class st_model(nn.Module):
  def __init__(self, num_features):
    super().__init__()
    self.layer1 = mnn.rLinear(in_features=num_features, out_features=1, weight_manifold=mnn.Stiefel)
  
  def forward(self, x):
    return self.layer1(x)

class sub_network(nn.Module):
  def __init__(self):
    super().__init__()
    self.layer1 = nn.Linear(1, 64)
    self.batchnorm1 = nn.BatchNorm1d(64)
    self.layer2 = nn.Linear(64, 256)
    self.batchnorm2 = nn.BatchNorm1d(256)
    self.layer3 = nn.Linear(256, 64)
    self.batchnorm3 = nn.BatchNorm1d(64)
    self.layer4 = nn.Linear(64, 1)
    self.batchnorm4 = nn.BatchNorm1d(1)

  def forward(self, x):
    x = F.relu(self.batchnorm1(self.layer1(x)))
    x = F.relu(self.batchnorm2(self.layer2(x)))
    x = F.relu(self.batchnorm3(self.layer3(x)))
    x = self.batchnorm4(self.layer4(x))
    return x


class model(nn.Module):
  def __init__(self, num_features):
    super().__init__()
    self.st_model1 = st_model(num_features)
    self.st_model2 = st_model(num_features)
    self.subnet1 = sub_network()
    self.subnet2 = sub_network()
    self.p1 = nn.Parameter(torch.rand(1))
    self.p2 = nn.Parameter(torch.rand(1))

  def forward(self, x):
    x1 = self.st_model1(x)
    x2 = self.st_model2(x)
    x1 = self.subnet1(x1)
    x2 = self.subnet2(x2)
    x = (self.p1 * x1) + (self.p2 * x2)
    return x


num_features=100
num_epochs=1000
learning_rate=0.01
model = model(num_features)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

for epoch in range(num_epochs):
    dataOutput = model(X)
    loss = criterion(dataOutput, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, num_epochs, loss))

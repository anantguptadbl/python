import pandas as pd
import numpy as np 
import os
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

torch.manual_seed(42)

class GCN(nn.Module):
    def __init__(self, feature_size, num_nodes):
        super().__init__()
        self.N = num_nodes
        self.W1 = torch.nn.Parameter(torch.rand(feature_size, 32))
        self.W2 = torch.nn.Parameter(torch.rand(32, 64))
        self.W3 = torch.nn.Parameter(torch.rand(64, 32))
        self.W4 = torch.nn.Parameter(torch.rand(32, feature_size))
        self.sig = nn.Sigmoid()
        self.b1 = nn.BatchNorm1d(32)
        self.b2 = nn.BatchNorm1d(64)
        self.b3 = nn.BatchNorm1d(32)
        self.b4 = nn.BatchNorm1d(feature_size)
        
    def forward(self, Z, A):
        Z = torch.matmul(torch.mm(A, Z), self.W1)
        Z = self.b1(Z)
        Z = self.sig(Z)
        self.embedding = torch.matmul(torch.mm(A, Z), self.W2)
        Z = self.b2(self.embedding)
        Z = self.sig(Z)
        Z = torch.matmul(torch.mm(A, Z), self.W3)
        Z = self.b3(Z)
        Z = self.sig(Z)
        Z = torch.matmul(torch.mm(A, Z), self.W4)
        Z = self.b4(Z)
        Z = self.sig(Z)
        return Z

# UNIT TEST
feature_size = 10
num_nodes = 100
adj = np.random.randint(2, size=(num_nodes, num_nodes))
features = np.random.randint(2, size=(num_nodes, feature_size))

model = GCN(feature_size, num_nodes)

# Config
criterion = nn.BCELoss()
learning_rate = 0.0001
num_epochs = 1000
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=learning_rate,
    weight_decay=1e-5
)

adj = Variable(torch.from_numpy(adj.astype(np.float32)))
features = Variable(torch.from_numpy(features.astype(np.float32)))

# Training
for epoch in range(num_epochs):
    dataOutput = model(features, adj)
    loss = criterion(dataOutput, features)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, num_epochs, loss))

# BASED on the following paper
# https://arxiv.org/abs/1901.03838

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import time

torch.manual_seed(42)

class singleNeuralNetwork(nn.Module):
    def __init__(self,inputDim,outputDim,k,T1,T2):
        super(singleNeuralNetwork,self).__init__()
        self.inputDim=inputDim
        self.outputDim=outputDim
        self.k=k
        self.w=Variable(torch.randn(k,inputDim,outputDim)).cuda()
        self.beta=Variable(torch.randn(k)).cuda()
        self.T1=T1
        self.T2=T2
        self.mu=Variable(torch.randn(outputDim)).cuda()
        
    def forward(self,x):
        self.w=torch.clamp(self.w,self.T1).cuda()
        self.beta=torch.clamp(self.beta,self.T2).cuda()
        results=torch.zeros(x.size()[0],self.outputDim).cuda()
        for curK in range(self.k):
            results= torch.add(results,(torch.matmul(x,self.w[curK]) * self.beta[curK])).cuda()
        return(results)
        

class xNN(nn.Module):
    def __init__(self,inputDim,linearOutputDim):
        super(xNN, self).__init__()
        self.inputDim=inputDim
        self.firstLayer=singleNeuralNetwork(inputDim,linearOutputDim,10,0.9,0.9).cuda()
        self.secondLayer=nn.Linear(linearOutputDim,2).cuda()
        
    def forward(self, x):
        x=self.firstLayer(x).cuda()
        x=self.secondLayer(x).cuda()
        return x

# Model Initialization
inputDim=100
linearOutputDim=20
outputDim=2
xNNModel=xNN(inputDim,linearOutputDim).cuda()
loss = torch.nn.MSELoss()
optimizer = optim.Adam(xNNModel.parameters(), lr=0.001,weight_decay=0.00001)

# Configuration
X=Variable(torch.randn(1000,inputDim)).cuda()
Y=Variable(torch.randn(1000,outputDim)).cuda()
epochs=1000

for curEpoch in range(epochs):
    xNNModel.zero_grad()
    output=xNNModel(X).cuda()
    curLoss=loss(output,Y)
    curLoss.backward()
    optimizer.step()
    if(curEpoch % 100==0):
        print("CurEpoch : {0}  Loss : {1}".format(curEpoch,curLoss.item()))

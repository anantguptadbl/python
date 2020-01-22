# The following is an example of HARD EXAMPLE MINING

# IMPORTS
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as data
import torchvision
from torchvision import transforms
import torchvision.models as models
import pickle
import torch.optim as optim

# Preparing the data
X=np.random.rand(10000,1024)
Y=np.zeros(10000)
Y[np.random.choice(10000,100)]=1
Y[np.random.choice(10000,50)]=2
Y[np.random.choice(10000,10)]=3

torch.manual_seed(42)

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.l1=nn.Linear(1024,256)
        self.b1=nn.BatchNorm1d(256)
        self.l2=nn.Linear(256,64)
        self.b2=nn.BatchNorm1d(64)
        self.l3=nn.Linear(64,8)
        self.b3=nn.BatchNorm1d(8)
        self.l4=nn.Linear(8,4)
        self.smax=nn.Softmax(dim=1)
        
    def forward(self,x):
        x=self.l1(x)
        x=self.b1(x)
        x=self.l2(x)
        x=self.b2(x)
        x=self.l3(x)
        x=self.b3(x)
        x=self.l4(x)
        x=self.smax(x)
        return(x)
    
# Config
epochs=20
batchSize=128
numBatches=int(Y.shape[0]/batchSize)

# Model Config
model=MyModel()
optimizer = optim.SGD(model.parameters(), lr=0.01,momentum=0.6)
criterion = nn.CrossEntropyLoss()

print("Total Batches are {0}".format(numBatches))
# Model Training
for epoch in range(epochs):     
    totalLoss=0
    curBatch=0
    while(1):
        model.zero_grad()
        output = model(Variable(torch.from_numpy(X[curBatch*batchSize:(curBatch+1)*batchSize].astype(np.float32))))      
        loss = criterion(output, Variable(torch.from_numpy(Y[curBatch*batchSize:(curBatch+1)*batchSize].astype(np.int))))   
        totalLoss=totalLoss + loss.item()         
        loss.backward()                 
        optimizer.step()
        if(epoch%1==0 and epoch>5):
          # We will reduce the data points of the class 0
          deselectIndices=np.where((output.detach().numpy()[:,0] < 0.80 ) & (Y[curBatch*batchSize:(curBatch+1)*batchSize]==0))
          deselectIndices=[x+(curBatch*batchSize) for x in deselectIndices][0]
          Y=Y[[x for x in range(X.shape[0]) if x not in deselectIndices]]
          X=X[[x for x in range(X.shape[0]) if x not in deselectIndices],:]
          numBatches=int(X.shape[0])
        curBatch=curBatch+1
        if(curBatch >= numBatches):
          break
        if(curBatch*batchSize > X.shape[0]):
          break
    if(epoch%1==0):
        print("Epoch {0}  TotalLoss {1}".format(epoch,totalLoss))



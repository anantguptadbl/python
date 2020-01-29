# NODE 2 VEC : GITHUB
# Imports
import pandas as pd
import numpy as np

# Creating the data
data=pd.DataFrame(np.random.rand(10000,100),columns=[str(x) for x in range(100)])
data['Node']=np.random.randint(100,10000)
data['target']=np.random.randint(10,10000)
uniqueNodesDict=dict((x,i) for i,x in enumerate(data['Node'].unique()))
featureNames=[str(x) for x in range(100)]

import os
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable

# Setting the manual seed
torch.manual_seed(42)

class Embedder(nn.Module):
    def __init__(self,numNodes,numEmbeddingDimensions,numFeatures):
        super(Embedder,self).__init__()
        # Creae the embeddings layer
        self.numEmbeddingDimensions=numEmbeddingDimensions
        self.embeddings=torch.randn(numNodes,numEmbeddingDimensions,requires_grad=True)
        self.l1=nn.Linear(numFeatures+numEmbeddingDimensions,256)
        self.r1=nn.BatchNorm1d(256)
        self.l2=nn.Linear(256,128)
        self.r2=nn.BatchNorm1d(128)
        self.l3=nn.Linear(128,64)
        self.r3=nn.BatchNorm1d(64)
        self.l4=nn.Linear(64,32)
        self.r4=nn.BatchNorm1d(32)
        self.l5=nn.Linear(32,1)
        
    def forward(self,inputs,cellIndices):
        inputs=torch.cat((inputs,self.embeddings[cellIndices,:].view(-1,self.numEmbeddingDimensions)), 1)
        output=self.l1(inputs)
        output=self.r1(output)
        output=self.l2(output)
        output=self.r2(output)
        output=self.l3(output)
        output=self.r3(output)
        output=self.l4(output)
        output=self.r4(output)
        output=self.l5(output)
        return(output)
 
# Model Configuration
batchSize=4096
inputDim=len(featureNames)
embeddingDim=50
epochs=1000
numNodes=len(uniqueNodesDict)


# Model Setup
model=Embedder(numNodes,embeddingDim,inputDim)
criterion=nn.MSELoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.00001)

numBatches=int(data.shape[0]/batchSize)
print(numBatches)
print("Starting the training")

optimizer=torch.optim.SGD(model.parameters(),lr=0.001,momentum=0.6)
for curEpoch in range(epochs):
    totalLoss=0
    for curBatch in range(numBatches):
        inpX=Variable(torch.from_numpy(data[featureNames].values[curBatch*batchSize:(curBatch+1)*batchSize,:].astype(np.float32)))
        inpY=Variable(torch.from_numpy(data['target'].values[curBatch*batchSize:(curBatch+1)*batchSize].astype(np.float32)))
        cellIndices=Variable(torch.from_numpy(np.array([uniqueNodesDict[y] for y in data['Node'][curBatch*batchSize:(curBatch+1)*batchSize]]).reshape(-1).astype(int))).type(torch.long)
        output=model(inpX,cellIndices)
        loss = criterion(output, inpY)
        totalLoss=totalLoss+loss.item()
        # We will perform the backward propagation
        loss.backward()
        optimizer.step()
        if(curEpoch % 1 == 0 and curBatch %25==0):
            print('epoch {0}, batch {1}, loss {2}'.format(curEpoch + 1, curBatch, totalLoss)) 

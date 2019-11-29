# Explanability with Artificial Neural Networks
import os
import numpy as np

import torch
from torch import nn
from torch.autograd import Variable

class linearRegressionOLS(nn.Module):
    def __init__(self):
        super(linearRegressionOLS,self).__init__()
        self.l1=nn.Linear(100,20)
        self.b1=nn.ReLU()
        self.l2=nn.Linear(20,10)
        self.b2=nn.ReLU()
        self.l3=nn.Linear(10,1)
        
    def forward(self,x):
        x = self.l1(x)
        x = self.b1(x)
        x = self.l2(x)
        x = self.b2(x)
        x = self.l3(x)
        return x
    
model=linearRegressionOLS()
criterion=nn.MSELoss()
optimizer=torch.optim.SGD(model.parameters(),lr=0.01)

X=np.random.rand(1000,100).astype(np.float32)
Y=np.random.randint(2,size=(1000)).reshape(1000,1).astype(np.float32)

numEpochs=10000
for epoch in range(numEpochs):
    dataInput=Variable(torch.from_numpy(X))
    outputVal=Variable(torch.from_numpy(Y))
    # In a gradient descent step, the following will now be performing the gradient descent now
    optimizer.zero_grad()
    # We will now setup the model
    dataOutput = model(dataInput)
    # We will now define the loss metric
    loss = criterion(dataOutput, outputVal)
    # We will perform the backward propagation
    loss.backward()
    optimizer.step()
    if epoch % 1000 == 0:
        print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, numEpochs, loss))      
        
layerNames=['l1','l2','l3']
layersDict={}
for layerName in layerNames:
    for featureNumber,feature in enumerate(model._modules[layerName].weight.detach().numpy()):
        curResult=Variable(torch.from_numpy(X[0].reshape(1,100)))
        for curLayer in model._modules:
            if(curLayer==layerName):
                break
            else:
                curResult=model._modules[curLayer](curResult)
        curResult=feature*curResult.detach().numpy()
        curResult=np.absolute(curResult)
        curResult=curResult/np.sum(curResult)
        if(layerName not in layersDict):
            layersDict[layerName]=[]
        layersDict[layerName].append(curResult)
        
for curLayerIndex in range(len(layerNames)-1):
    for curSubLayer in layersDict[layerNames[len(layerNames)-curLayerIndex-1]]:   
        for i,x in enumerate(curSubLayer):
            layersDict[layerNames[len(layerNames)-curLayerIndex-2]][i]=layersDict[layerNames[len(layerNames)-curLayerIndex-2]][i]*x[0]
featureImportance=np.sum(np.array(layersDict[layerNames[0]]),axis=0)

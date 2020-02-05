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

print("Import Cell Execution Completed")

# Fixed Data Set
from sklearn.datasets import load_boston
X, y = load_boston(return_X_y=True)

# We will adjust y
y=np.array([0 if x<10 else 1 for x in y])

# We will normalize the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X=scaler.fit_transform(X)

# Train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Simple Model

torch.manual_seed(42)

class Model1(nn.Module):
  def __init__(self):
    super(Model1,self).__init__()
    self.l1=nn.Linear(13,10)
    self.b1=nn.BatchNorm1d(10)
    self.l2=nn.Linear(10,5)
    self.b2=nn.BatchNorm1d(5)
    self.l3=nn.Linear(5,1)
    self.sigm=nn.Sigmoid()

  def forward(self,inputs):
    output=F.relu(self.b1(self.l1(inputs)))
    output=F.relu(self.b2(self.l2(output)))
    output=self.l3(output)
    output=self.sigm(output)
    return(output)

model=Model1()
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.BCELoss()

numEpochs=4000
batchSize=32
numBatches=int(X_train.shape[0]/batchSize)

for curEpoch in range(numEpochs):
  totalLoss=0
  for curBatch in range(numBatches):
    model.zero_grad()
    modelOutput=model(Variable(torch.from_numpy(X_train[curBatch*batchSize:(curBatch+1)*batchSize].astype(np.float32))))
    loss=criterion(modelOutput,Variable(torch.from_numpy(y_train[curBatch*batchSize:(curBatch+1)*batchSize].astype(np.float32))))
    totalLoss=totalLoss + loss.item()
    loss.backward()
    optimizer.step()
  if(curEpoch%1000==0):
    print("Epoch {0} Loss is {1}".format(curEpoch,totalLoss))

# Accuracy on test data
model.zero_grad()
modelOutput=model(Variable(torch.from_numpy(X_test.astype(np.float32))))
loss=criterion(modelOutput,Variable(torch.from_numpy(y_test.astype(np.float32))))
print("Test Loss is {0}".format(loss))
# Test Loss is 0.7346357703208923

# I will try to quantize this using my own function

torch.manual_seed(42)

def quantize(x,minVal,maxVal):
  # For the array x we will first find the bins
  # and then quantize
  bins=np.linspace(minVal,maxVal,10)
  ind = np.digitize(x.detach().numpy(), bins)
  return((bins,ind))

class ModelWrapper(nn.Module):
  def __init__(self,baseModel):
    super(ModelWrapper,self).__init__()
    self.bins=[]
    self.ind=[]
    self.layer=baseModel
    self.weightSize=self.layer.weight.size()

  def forward(self,inputs):
    #print("Reached forward of modelWrapper and the size of inputs is {0}".format(inputs.size()))
    output=self.layer(inputs)
    #print("The output within the forward is {0}".format(output.size()))
    weightShape=self.layer.weight.size()
    #print("self.layer.weight is {0}".format(self.layer.weight.view(-1).size()))
    minVal=torch.min(self.layer.weight.view(-1))
    #print("minVal is {0}".format(minVal))
    maxVal=torch.max(self.layer.weight.view(-1))
    #print("maxVal is {0}".format(maxVal))
    #print("Reached the point where we will be setting the bins and the ind in the inidvidual layers")
    self.bins,self.ind=quantize(self.layer.weight.view(-1).cpu(),minVal.cpu().detach().numpy(),maxVal.cpu().detach().numpy())
    return(output)

class Model1(nn.Module):
  def __init__(self):
    super(Model1,self).__init__()
    self.bins=[]
    self.ind=[]
    self.l1=ModelWrapper(nn.Linear(13,10))
    self.l2=ModelWrapper(nn.Linear(10,5))
    self.l3=ModelWrapper(nn.Linear(5,1))
    self.b1=nn.BatchNorm1d(10)
    self.b2=nn.BatchNorm1d(5)
    self.sigm=nn.Sigmoid()
    
  def forward(self,inputs):
    # For each input feature, we will do min max
    for curInputFeature in range(inputs.size()[1]):
      if(len(self.bins)==curInputFeature):
        minVal=torch.min(inputs[:,curInputFeature])
        maxVal=torch.max(inputs[:,curInputFeature])
        bins,ind=quantize(inputs[:,curInputFeature],minVal,maxVal)
        self.bins.append(bins)
        self.ind.append(ind)
      else:
        minVal=np.min(np.array(list(inputs[:,curInputFeature].cpu().numpy().reshape(-1)) + list(self.bins[curInputFeature])))
        maxVal=np.max(np.array(list(inputs[:,curInputFeature].cpu().numpy().reshape(-1)) + list(self.bins[curInputFeature])))
        bins,ind=quantize(inputs[:,curInputFeature],minVal,maxVal)
        self.bins[curInputFeature]=bins
        self.ind[curInputFeature]=ind
    output=F.relu(self.b1(self.l1(inputs)))
    output=F.relu(self.b2(self.l2(output)))
    output=self.l3(output)
    output=self.sigm(output)
    return(output)

model=Model1()
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.BCELoss()

numEpochs=10000
batchSize=128
numBatches=int(X_train.shape[0]/batchSize)
print("Num batches is {0}".format(numBatches))

for curEpoch in range(numEpochs):
  totalLoss=0
  for curBatch in range(numBatches):
    model.zero_grad()
    modelOutput=model(Variable(torch.from_numpy(X_train[curBatch*batchSize:(curBatch+1)*batchSize].astype(np.float32))))
    loss=criterion(modelOutput,Variable(torch.from_numpy(y_train[curBatch*batchSize:(curBatch+1)*batchSize].astype(np.float32))))
    totalLoss=totalLoss + loss.item()
    loss.backward()
    optimizer.step()
  if(curEpoch%10==0):
    print("Epoch {0} Loss is {1}".format(curEpoch,totalLoss))

# Setting the weights based on quantization
model._modules['l1'].weight=torch.FloatTensor(np.array([model._modules['l1'].bins[x-1] for x in model._modules['l1'].ind]).reshape(model._modules['l1'].weightSize[0],model._modules['l1'].weightSize[1]))
model._modules['l2'].weight=torch.FloatTensor(np.array([model._modules['l2'].bins[x-1] for x in model._modules['l2'].ind]).reshape(model._modules['l2'].weightSize[0],model._modules['l2'].weightSize[1]))
model._modules['l3'].weight=torch.FloatTensor(np.array([model._modules['l3'].bins[x-1] for x in model._modules['l3'].ind]).reshape(model._modules['l3'].weightSize[0],model._modules['l3'].weightSize[1]))

# Accuracy on test data
model.zero_grad()
modelOutput=model(Variable(torch.from_numpy(X_test.astype(np.float32))))
loss=criterion(modelOutput,Variable(torch.from_numpy(y_test.astype(np.float32))))
print("Test Loss is {0}".format(loss))

#Test Loss is 0.7079697847366333


import os
from scipy import misc
import glob
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms

class simpleImageClassficationCNN(nn.Module):
    def __init__(self):
        super(simpleImageClassficationCNN,self).__init__()
        self.outChannels=16
        self.kernelSizeConv=4
        self.kernelSizeMaxPool=2
        self.linearOutput=64
        self.finalOutput=10
        self.convHOutput=28 -(self.kernelSizeConv-1)
        self.maxHOutput=self.convHOutput -(self.kernelSizeMaxPool-1)
        self.conv1 = torch.nn.Conv2d(4, self.outChannels, kernel_size=self.kernelSizeConv, stride=1, padding=0)
        self.conv1_bn = nn.BatchNorm2d(self.outChannels)
        self.pool = torch.nn.MaxPool2d(kernel_size=self.kernelSizeMaxPool, stride=1, padding=0)
        self.fc1 = torch.nn.Linear(self.maxHOutput*self.maxHOutput*self.outChannels, self.linearOutput)
        self.dense1_bn = nn.BatchNorm1d(self.linearOutput)
        self.fc2 = torch.nn.Linear(self.linearOutput, self.finalOutput)
        
    def forward(self, x):
        #Computes the activation of the first convolution
        #Size changes from (4, 28, 28) to (outChannels, convHOutput, convHOutput)
        x = F.relu(self.conv1_bn(self.conv1(x)))
        
        #Size changes from (outChannels, convHOutput, convHOutput) to (outChannels, maxHOutput, maxHOutput)
        x = self.pool(x)
        
        # We will now be perfoming Linear Neural Nets, so we need to convert the data to a single dimension
        #Size changes from (outChannels, maxHOutput, maxHOutput) to (1, outChannels * maxHOutput * maxHOutput)
        x = x.view(-1, self.outChannels * self.maxHOutput * self.maxHOutput)
        
        #Computes the activation of the first fully connected layer
        #Size changes from (1, outChannels * maxHOutput * maxHOutput) to (1, linearOutput)
        x = F.relu(self.dense1_bn(self.fc1(x)))
        
        #Computes the second fully connected layer (activation applied later)
        #Size changes from (1, linearOutput) to (1, finalOutput)
        x = self.fc2(x)
        return(x)
        
  data=[]
import random

for x in range(1,20001):
    data.append(np.reshape(misc.imread('train/{}.png'.format(x)),(4*28*28)))
print("The data has been loaded")

import pandas as pd
trainLabel=pd.read_csv('train.csv')
for x in range(10):
    trainLabel[x]=trainLabel['label'].map(lambda y : 1 if y==x else 0)

learning_rate=0.001
num_epochs=200
model = simpleImageClassficationCNN()
#criterion = nn.MSELoss()
criterion = nn.CrossEntropyLoss()
#criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

imageBatchSize=100

def cross_entropy(pred, soft_targets):
    logsoftmax = nn.LogSoftmax()
    return torch.mean(torch.sum(- pred  * logsoftmax(soft_targets), 1))

for epoch in range(num_epochs):
    random.shuffle(data)
    totalBatchLoss=0
    for x in range(len(data)/imageBatchSize):
        dataInput=Variable(torch.from_numpy(np.array(data[imageBatchSize *x : imageBatchSize * (x+1)]).reshape(imageBatchSize,4,28,28).astype(np.float32)))
        dataOutput = model(dataInput)
        # 10 CLASSES
        origDataOutput=Variable(torch.from_numpy(trainLabel[range(10)].values[x].reshape(1,10)))
        #origDataOutput = torch.LongTensor(trainLabel[range(10)].values[x].reshape(1,10))
        #origDataOutput=Variable(torch.from_numpy(trainLabel[trainLabel['id']==x]['label'].values.astype(np.float32)))
        # DOUBLE
        #origDataOutput=Variable(torch.from_numpy(trainLabel['label'].values[imageBatchSize *x : imageBatchSize * (x+1)].astype(np.float32).reshape(imageBatchSize,1)))
        # LONG
        #origDataOutput=Variable(torch.from_numpy(trainLabel['label'].values[imageBatchSize *x : imageBatchSize * (x+1)].astype(np.long).reshape(imageBatchSize,1)))
        #loss = criterion(dataOutput,origDataOutput)
        #loss = -torch.mean(torch.sum(torch.sum(torch.sum(origDataOutput * torch.log(dataOutput), axis=1), axis=1), axis=1))
        #print(origDataOutput)
        #print(dataOutput)
        loss = cross_entropy(origDataOutput.type(torch.DoubleTensor),dataOutput.type(torch.DoubleTensor))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        totalBatchLoss=totalBatchLoss+loss
    if epoch % 10 == 0:
        print('For EPOCH [{}/{}] the total batch loss is {}'.format(epoch + 1, num_epochs, totalBatchLoss))

#### import pandas as pd
import numpy as np
import imageio
import os
import matplotlib.pyplot as plt
%matplotlib inline

# TRAINING DATA
imageData=np.array([])
imageY=[]
for curShape in ['circle','square','star','triangle']:
    print("CurShape is {0}".format(curShape))
    for curFile in os.listdir("../input/shapes/{0}".format(curShape))[0:500]:
        if(imageData.shape[0] ==0):
            imageData=imageio.imread("../input/shapes/{0}/{1}".format(curShape,curFile)).reshape(200,200,1)
        else:
            imageData=np.append(imageData,imageio.imread("../input/shapes/{0}/{1}".format(curShape,curFile)).reshape(200,200,1),axis=0)
        imageY.append(curShape)
imageData=imageData.reshape(-1,1,200,200).astype(np.float32)
imageY=np.array(imageY)        
imageY=[[1,0,0,0] if x=='circle' else [0,1,0,0] if x=='square' else [0,0,1,0] if x=='star' else [0,0,0,1] for x in imageY]
imageY=np.array(imageY).astype(np.float32)
print("Final imageData shape is {0} and that of imageY is {1}".format(imageData.shape,imageY.shape)) # 200,200

# TEST DATA
imageTestData=np.array([])
imageTestY=[]
for curShape in ['circle','square','star','triangle']:
    print("CurShape is {0}".format(curShape))
    for curFile in os.listdir("../input/shapes/{0}".format(curShape))[500:600]:
        if(imageTestData.shape[0] ==0):
            imageTestData=imageio.imread("../input/shapes/{0}/{1}".format(curShape,curFile)).reshape(200,200,1)
        else:
            imageTestData=np.append(imageTestData,imageio.imread("../input/shapes/{0}/{1}".format(curShape,curFile)).reshape(200,200,1),axis=0)
        imageTestY.append(curShape)
imageTestData=imageTestData.reshape(-1,1,200,200).astype(np.float32)
imageTestY=np.array(imageTestY)
imageTestY=[[1,0,0,0] if x=='circle' else [0,1,0,0] if x=='square' else [0,0,1,0] if x=='star' else [0,0,0,1] for x in imageTestY]
imageTestY=np.array(imageTestY).astype(np.float32)
print("Final imageTestData shape is {0} and that of imageTestY is {1}".format(imageTestData.shape,imageTestY.shape)) # 200,200


import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=4,kernel_size=20,stride=1,padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
        self.conv2 = nn.Conv2d(in_channels=4,out_channels=1,kernel_size=10,stride=3,padding=0)
        self.fc1 = nn.Linear(13*13*1, 8)
        self.fc2 = nn.Linear(8, 4)
        self.smax = nn.Softmax()

    def forward(self, x):
        #print("Forward Function Entry Size : {0}".format(x.size()))
        #x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv1(x)))
        #print("Forward Function Conv1 and Pool : {0}".format(x.size()))
        x = self.pool(F.relu(self.conv2(x)))
        #print("Forward Function Conv2 and Pool : {0}".format(x.size()))
        x = x.view(-1, 13 * 13 * 1)
        x = F.relu(self.fc1(x))
        #print("Forward Function fc1 : {0}".format(x.size()))
        x = self.fc2(x)
        # Applying softmax
        x = self.smax(x)
        return x

model=Net()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()
epochRange=30
batches=8
batchSize=50

# Choice List
choiceBatches=[]
for curBatch in range(batches):
    choiceBatches.append(random.choices(list(range(imageData.shape[0])), k=batchSize))

import random
for epoch in range(epochRange):
    totalLoss=0
    for curBatch in range(batches):
        choiceList=choiceBatches[curBatch]
        inputs=Variable(torch.from_numpy(imageData[choiceList,:,:,:]))
        #print("Size of inputs is {0}".format(inputs.size()))
        labels=Variable(torch.from_numpy(np.array(imageY)[[choiceList]]))
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = model(inputs)
        #print("Before entering the loss criterion outputs={0} and labels={1}".format(outputs.size(),labels.size()))
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # print statistics
        totalLoss += loss.item()
    if(epoch % 1==0):
        print("Epoch : {0} Total Loss : {1}".format(epoch,totalLoss))
        
# Prediction
predictions=model(Variable(torch.from_numpy(imageTestData)))

# Prediction Metrics
def getData(x):
    y=[0,0,0,0]
    y[np.argmax(x)]=1
    return y

predictions=[getData(x) for x in predictions.detach()]
incorrectCount=0
# Getting the accuracy
for x in range(len(predictions)):
    if((imageTestY[x]!= predictions[x]).any()):
        incorrectCount=incorrectCount+1
        
print("Accuracy is {0}".format(1-(incorrectCount/len(predictions))))

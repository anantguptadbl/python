# Read the classes
import json
import pandas as pd
import numpy as np


with open('/home/anantgupta/Documents/Programming/deepLearning/YOLO/annotations/instances_val2017.json') as f:
    data=json.load(f)
    

data=pd.DataFrame([[x['image_id'],1,x['iscrowd'],x['bbox']] for x in data['annotations'] if x['category_id']==1] + [[x['image_id'],0,0,[]] for x in data['annotations'] if x['category_id']!=1],columns=['imageId','class','crowdFlag','bbox'])
data['bbox']=data['bbox'].map(lambda x : tuple(x))
data=data[['imageId','class','crowdFlag','bbox']].drop_duplicates()
data['bbox']=data['bbox'].map(lambda x : 0 if len(x)==0 else x)
data=data[data['class']==1]
#data['bbox']=data['bbox'].map(lambda x : list(x))

import cv2
listImageIds=data['imageId'].unique()

print("Cell Execution Completed")
# Read Images 

# We are using the COCO dataset as it is without breaking it further into sub images
def getData(curImageId,data):
    trainingY=[]
    trainingX=[]
    curImage=cv2.imread("./val2017/{:012d}.jpg".format(curImageId))
    curImageX,curImageY=curImage.shape[0],curImage.shape[1]
    curImage=cv2.resize(curImage,(300,300))
    #if(curImage.shape[0]!=640):
    #    curImage=cv2.resize(curImage,(640,480))
    #    print(curImage.shape)
    curData=data[data['imageId']==curImageId]
    curData['centreX']=curData['bbox'].map(lambda x : (x[0] + x[2])/2 if x !=0 else 0)
    curData['centreY']=curData['bbox'].map(lambda x : (x[1] + x[3])/2 if x !=0 else 0)
    trainingYSubSet=[]
    boundingData=curData
    if(boundingData.shape[0] >= 3):
        boundingData=boundingData.head(3)
        leftOver=0
    if(boundingData.shape[0] < 3):
        leftOver=3-boundingData.shape[0]
    if(boundingData.shape[0] > 0):
        trainingYSubSet.append(1)
        for curRow in boundingData.values:
            trainingYSubSet.extend([curRow[1],curRow[3][0]*(300/curImageY),curRow[3][1]*(300/curImageX),curRow[3][2]*(300/curImageY),curRow[3][3]*(300/curImageX)])    
        for curRow in range(leftOver):
            trainingYSubSet.extend([0,0,0,0,0])
        trainingY.append(trainingYSubSet)
        trainingX.append(curImage)
    return([trainingX,trainingY,curImage])

trainingX=[]
trainingY=[]
for curImage in listImageIds:
    results=getData(curImage,data)
    trainingX.extend(results[0])
    trainingY.extend(results[1])

print("Converting to numpy array")
trainingX=np.array(trainingX)
trainingY=np.array(trainingY)

np.save("trainingX",trainingX)
np.save("trainingY",trainingY)

print("Cell Execution Completed")

# Normalizing the Bounding Boxes for all the anchor points
trainingX=np.load("trainingX.npy")
trainingY=np.load("trainingY.npy")
trainingY[:,2:6]=trainingY[:,2:6]/300
trainingY[:,7:11]=trainingY[:,7:11]/300
trainingY[:,12:16]=trainingY[:,12:16]/300
print(trainingX.shape)
print(trainingY.shape)

# Original YOLO Paper
# https://arxiv.org/pdf/1506.02640v5.pdf

import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import time

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.fastest = False
torch.manual_seed(42)

class YOLOModel(nn.Module):
    def __init__(self):
        super(YOLOModel,self).__init__()
        self.B=5
        self.C=1
        self.S=1
        self.conv1=nn.Conv2d(3,8,3, stride=1, padding=0).cuda()
        #self.b1=nn.BatchNorm2d(32, 0.8).cuda()
        self.max1=nn.MaxPool2d(2, stride=2).cuda()
        self.conv2=nn.Conv2d(8,16,3, stride=1, padding=0).cuda()
        #self.b1=nn.BatchNorm2d(64, 0.8)
        self.max2=nn.MaxPool2d(2, stride=2).cuda()
        self.conv3=nn.Conv2d(16,32,3, stride=1, padding=0).cuda()
        self.max3=nn.MaxPool2d(2, stride=2)
        self.conv4=nn.Conv2d(32,64,3, stride=1, padding=0)
        #self.conv4=nn.Conv2d(1024,1024,3, stride=1, padding=0)
        #self.b1=nn.BatchNorm2d(64, 0.8)
        self.max4=nn.MaxPool2d(2, stride=1)
        #self.conv5=nn.Conv2d(64,64,3, stride=1, padding=0)
        self.conv5=nn.Conv2d(64,128,3, stride=1, padding=0)
        self.conv6=nn.Conv2d(128,256,3, stride=1, padding=0)
        self.max5=nn.MaxPool2d(2, stride=2)
        self.conv7=nn.Conv2d(256,512,3, stride=1, padding=0)
        self.max6=nn.MaxPool2d(2, stride=2)
        self.conv8=nn.Conv2d(512,1024,2, stride=1, padding=0)
        self.l1=nn.Linear(1024*5*5,16).cuda()
        self.sigm1=nn.Sigmoid().cuda()
        
    
    def forward(self,x):
        x=self.conv1(x).cuda()
        x=self.max1(x).cuda()
        x=self.conv2(x).cuda()
        x=self.max2(x).cuda()
        x=self.conv3(x).cuda()
        x=self.max3(x).cuda()
        x=self.conv4(x).cuda()
        x=self.max4(x).cuda()
        x=self.conv5(x).cuda()
        x=self.conv6(x).cuda()
        x=self.max5(x).cuda()
        x=self.conv7(x).cuda()
        x=self.max6(x).cuda()
        x=self.conv8(x).cuda()
        x=x.view(-1,1024*5*5).cuda()
        x=self.l1(x).cuda()
        x=self.sigm1(x)
        return(x)
        
torch.cuda.empty_cache()

batchSize=32
batches=int(trainingX.shape[0]/batchSize)

model=YOLOModel().cuda()
criterion1=nn.MSELoss().cuda()
criterion2=nn.MSELoss().cuda()
criterion5=nn.BCELoss().cuda()
criterion6=nn.BCELoss().cuda()
criterion7=nn.BCELoss().cuda()
criterion8=nn.BCELoss().cuda()
def lossCalc2(x,y):
    x=x[:,[4,5,9,10,14,15]]
    y=y[:,[4,5,9,10,14,15]]
    x[x < 0] = 0
    y[y < 0] = 0
    x=torch.sqrt(x)
    y=torch.sqrt(y)
    #print(x)
    #print(y)
    return(criterion2(x,y))

print("Cell Execution Completed")

#torch.save(model,"Model_Large1")
#model=torch.load("Model_Small_16Batches")
batchSize=16
batches=int(trainingX.shape[0]/batchSize)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

startTime=time.time()
for epoch in range(100000):
    totalLoss=0
    for curBatch in range(batches):
        model.zero_grad()
        X=Variable(torch.from_numpy(trainingX[curBatch*batchSize:(curBatch+1)*batchSize].reshape(batchSize,3,300,300).astype(np.float32))).cuda()
        Y=Variable(torch.from_numpy(trainingY[curBatch*batchSize:(curBatch+1)*batchSize].astype(np.float32))).cuda()
        XOut=model(X)
        loss1=criterion1(XOut[:,[2,3,7,8,12,13]],Y[:,[2,3,7,8,12,13]])
        loss2=criterion2(XOut[:,[4,5,9,10,14,15]],Y[:,[4,5,9,10,14,15]])
        loss3=criterion5(XOut[:,0],Y[:,0])
        loss4=criterion6(XOut[:,1],Y[:,1])
        loss5=criterion7(XOut[:,6],Y[:,6])
        loss6=criterion8(XOut[:,11],Y[:,11])
        loss= loss1 + loss2 + loss3 + loss4 + loss5 + loss6
        loss.backward()
        optimizer.step()
        totalLoss=totalLoss + loss.item()
    if(epoch %1==0):
        print("Epoch: {0} Loss is {1} TimeTaken is {2}".format(epoch,totalLoss,time.time()-startTime))
    if((epoch %100==0) & (epoch > 1)):
        print("Saving")
        torch.save(model,"Model_Large1")
        
# TESTING
print(trainingY[1])
print(model(Variable(torch.from_numpy(trainingX[1:2].reshape(1,3,300,300).astype(np.float32))).cuda()))
    

# Read the classes
import json
import pandas as pd
import numpy as np


with open('./annotations/instances_val2017.json') as f:
    data=json.load(f)
    

data=pd.DataFrame([[x['image_id'],1,x['iscrowd'],x['bbox']] for x in data['annotations'] if x['category_id']==1] + [[x['image_id'],0,0,[]] for x in data['annotations'] if x['category_id']!=1],columns=['imageId','class','crowdFlag','bbox'])
data['bbox']=data['bbox'].map(lambda x : tuple(x))
data=data[['imageId','class','crowdFlag','bbox']].drop_duplicates()
data['bbox']=data['bbox'].map(lambda x : 0 if len(x)==0 else x)
data=data[data['class']==1]
#data['bbox']=data['bbox'].map(lambda x : list(x))

import cv2
  
print("Cell Execution Completed")
# Read Images 

trainingY=[]
trainingX=[]
for curImageId in data['imageId'].unique(): 
    #print("./val2017/{:012d}.jpeg".format(curImageId))
    curImage=cv2.imread("./val2017/{:012d}.jpg".format(curImageId))
    curData=data[data['imageId']==curImageId]
    curData['centreX']=curData['bbox'].map(lambda x : x[0] + x[2]/2 if x !=0 else 0)
    curData['centreY']=curData['bbox'].map(lambda x : x[1] + x[3]/2 if x !=0 else 0)

    for boundX in range(4):
        for boundY in range(4):
            trainingYSubSet=[]
            boundingData=curData[(curData['centreX'] < (boundX+1)*(640/4) ) & (curData['centreX'] >= boundX*(640/4)) & (curData['centreY'] < (boundY+1)*(480/4))  & (curData['centreY'] >= boundY*(480/4))]
            boundingData=boundingData[boundingData['bbox'] != 0]
            if(boundingData.shape[0] >= 3):
                boundingData=boundingData.head(3)
                leftOver=0
            if(boundingData.shape[0] < 3):
                leftOver=3-boundingData.shape[0]

            if(boundingData.shape[0] > 0):
                trainingYSubSet.append(1)
                for curRow in boundingData.values:
                    trainingYSubSet.extend([curRow[1],curRow[3][0],curRow[3][1],curRow[3][2],curRow[3][3]])
            else:
                trainingYSubSet.append(0)

            for curRow in range(leftOver):
                trainingYSubSet.extend([0,0,0,0,0])
                
            trainingY.append(trainingY)
            trainingX.append(curImage[int(boundX*(640/4)):int((boundX+1)*(640/4)),int(boundY*(480/4)):int((boundY+1)*(480/4)),:])

print("Converting to numpy array")
trainingX=np.array(trainingX)
trainingY=np.array(trainingY)

print("Cell Execution Completed")
#boundingData

# Original YOLO Paper
# https://arxiv.org/pdf/1506.02640v5.pdf
class YOLOModel(nn.Module):
    def __init__(self):
        self.B=5
        self.C=2
        self.S=3
        self.conv1=nn.Conv2d(3,64,7, stride=2, padding=0)
        self.b1=nn.BatchNorm2d(64, 0.8)
        self.max1=nn.MaxPool2d(2, stride=2)
        self.conv2=nn.Conv2d(64,192,3, stride=0, padding=0)
        #self.b1=nn.BatchNorm2d(64, 0.8)
        self.max2=nn.MaxPool2d(2, stride=2)
        self.conv3=nn.Conv2d(192,256,3, stride=0, padding=0)
        #self.b1=nn.BatchNorm2d(64, 0.8)
        self.max3=nn.MaxPool2d(2, stride=2)
        self.conv4=nn.Conv2d(256,512,3, stride=0, padding=0)
        #self.b1=nn.BatchNorm2d(64, 0.8)
        self.max5=nn.MaxPool2d(2, stride=2)
        self.conv6=nn.Conv2d(256,512,3, stride=0, padding=0)
        #self.b1=nn.BatchNorm2d(64, 0.8)
        self.max6=nn.MaxPool2d(2, stride=2)
        self.conv7=nn.Conv2d(512,1024,3, stride=0, padding=0)
        #self.b1=nn.BatchNorm2d(64, 0.8)
        self.max7=nn.MaxPool2d(2, stride=2)
    
    def forward(self,x):
        x=conv1(x)
        x=b1(x)
        x=max1(x)
        x=conv2(x)
        x=max2(x)
        x=conv3(x)
        x=max3(x)
        x=conv4(x)
        x=max4(x)
        x=conv5(x)
        x=max5(x)
        x=conv6(x)
        x=max6(x)
        x=conv7(x)
        x=max7(x)
        print("Size of x is {0}".format(x.size()))
        
model=YOLOModel()
criterion=nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

for epoch in range(1):
    inputData
        
        
        
        
        

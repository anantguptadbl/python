import pandas as pd
import numpy as np
import nltk
data=pd.read_csv('quora-question-pairs/train.csv')

data['question1']=data['question1'].map(lambda x : str(x).lower().replace('?','').replace('-','').replace('(','').replace(')','').replace('[','').replace(']',''))
data['question2']=data['question2'].map(lambda x : str(x).lower().replace('?','').replace('-','').replace('(','').replace(')','').replace('[','').replace(']',''))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

vectorizer = TfidfVectorizer()
transformer =TruncatedSVD(100)

X = vectorizer.fit_transform(data['question1'].values)
X = transformer.fit_transform(X)
data['question1_vectorized']=X.tolist()

vectorizer = TfidfVectorizer()
transformer =TruncatedSVD(100)

X = vectorizer.fit_transform(data['question2'].values)
X = transformer.fit_transform(X)
data['question2_vectorized']=X.tolist()

q1=np.array(data['question1_vectorized'].values.tolist()).astype(np.float32)
q2=np.array(data['question2_vectorized'].values.tolist()).astype(np.float32)
labels=data['is_duplicate'].values.astype(np.float32)
del(data)

print("Cell Execution Completed")

import os
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

torch.manual_seed(42)

class ContrastiveLoss(nn.Module):
    def __init__(self,margin=1.0):
        super(ContrastiveLoss,self).__init__()
        self.margin=margin
        
    def forward(self,input1,input2,label):
        dW=F.pairwise_distance(input1,input2).cuda()
        contrastiveLoss=torch.mean( (1-label)* torch.pow(dW,2) + label*(torch.pow(torch.clamp(self.margin - dW, min=0.0), 2))).cuda()
        return(contrastiveLoss)

class netLayer(nn.Module):
    def __init__(self):
        super(netLayer,self).__init__()
        self.l1=nn.Linear(100,200).cuda()
        self.l2=nn.Linear(200,50).cuda()
        self.l3=nn.Linear(50,10).cuda()
    
    def forward(self,x):
        x=self.l1(x).cuda()
        x=self.l2(x).cuda()
        x=self.l3(x).cuda()
        return(x)
    
class siamese(nn.Module):
    def __init__(self):
        super(siamese,self).__init__()
        self.net=netLayer()

    def forward(self,x,y):
        out1=self.net(x).cuda()
        out2=self.net(y).cuda()
        return(out1,out2)
    

num_epochs=10000
numBatches=int(q1.shape[0]/batchSize)
batchSize=128
model=siamese().cuda()
criterion=ContrastiveLoss().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)

for epoch in range(num_epochs):
    totalLoss=0
    for curBatch in range(numBatches):
        model.zero_grad()
        curInput1=Variable(torch.from_numpy(q1[curBatch*batchSize:(curBatch+1)*batchSize])).cuda()
        curInput2=Variable(torch.from_numpy(q2[curBatch*batchSize:(curBatch+1)*batchSize])).cuda()
        curOutput=Variable(torch.from_numpy(labels[curBatch*batchSize:(curBatch+1)*batchSize])).cuda()
        modelOutput1,modelOutput2=model(curInput1,curInput2)
        loss=criterion(modelOutput1,modelOutput2,curOutput)
        loss.backward()
        optimizer.step()
        totalLoss=totalLoss + loss.item()
    if(epoch%10==0):
        print("Epoch {0}  Batch{1} Loss {2}".format(epoch,curBatch,totalLoss))

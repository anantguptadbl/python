import pandas as pd
import numpy as np

data=pd.read_csv('/home/ubuntu/anant/projects/ncml/ml-20m/ratings.csv')
data=data[data['movieId'] <= 1000]
from datetime import datetime
data['yearmonth']=data['timestamp'].map(lambda x : str(datetime.utcfromtimestamp(x).year) + str(datetime.utcfromtimestamp(x).month)  )
data=data[data['yearmonth'].isin(['20151','20152','20153','201412','201411','201410'])]

def getOneHot(x):
    a=np.zeros(1000)
    a[x-1]==1
    return(a)

data['movieCoding']=data['movieId'].map(lambda x : getOneHot(x))
movieEncodingLength=max(data['movieId'].values)+1

def getData(curData):
    a=np.zeros(movieEncodingLength)
    for y in curData.values:
        a[int(y[0])-1]=int(y[1])
    return a        

results=data.groupby(['userId','yearmonth'],as_index=False)['movieId','rating'].apply( lambda x : pd.Series({'data':getData(x)})).reset_index()
results=results.pivot(index='userId', columns='yearmonth', values='data').fillna(0)
results=results.applymap(lambda x : np.zeros(movieEncodingLength) if type(x)==int else x)
rows,columns=results.values.shape

results1=results[['201410','201411','201412','20151','20152']].values.reshape(rows*(columns-1),1)
results1=results1[0:2000]

results2=results[['201411','201412','20151','20152','20153']].values.reshape(rows*(columns-1),1)
results2=results2[0:2000]

inpData=[]
for x in results1:
    inpData.append(x[0])
inpData=np.vstack(inpData)

outData=[]
for x in results2:
    outData.append(x[0])
outData=np.vstack(outData)


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class LSTMSimple(nn.Module):
    def __init__(self,inputDim,hiddenDim,outputDim,batch_size,step_size):
        super(LSTMSimple,self).__init__()
        self.inputDim=inputDim
        self.hiddenDim=hiddenDim
        self.outputDim=outputDim
        self.batch_size=batch_size
        self.step_size=step_size
        torch.manual_seed(1)
        # input_size – The number of expected features in the input x
        # hidden_size – The number of features in the hidden state h
        # num_layers – Number of recurrent layers
        self.lstm=nn.LSTM(self.inputDim,self.hiddenDim)
        # Hidden state is a tuple of two states, so we will have to initialize two tuples
        # h_0 of shape (num_layers * num_directions, batch, hidden_size)
        # c_0 of shape (num_layers * num_directions, batch, hidden_size)
        self.hidden = (torch.randn(1,self.batch_size,self.hiddenDim) , torch.rand(1,self.batch_size,self.hiddenDim))
        self.linearModel=nn.Linear(self.hiddenDim,self.outputDim)

    def forward(self,inputs):
        self.fullDataOutput=[]
        # input of shape (seq_len, batch, input_size)
        # h_0 of shape (num_layers * num_directions, batch, hidden_size)
        # c_0 of shape (num_layers * num_directions, batch, hidden_size)
        self.out,self.hidden = self.lstm(inputs,self.hidden)
        self.outLinear=self.linearModel(self.out)
        self.fullDataOutput.append(self.outLinear)
        return torch.stack(self.fullDataOutput).view(-1,self.outputDim)
        
def lossCalc(x,y):
    return torch.sum(torch.add(x,-y)).pow(2) 

# Model Object
inputDim=1000
hiddenDim=100
outputDim=1000

# LSTM Configuration
totalElements=len(inpData)
batch_size=10
step_size=5
totalBatches=totalElements/(step_size*batch_size)


model=LSTMSimple(inputDim,hiddenDim,outputDim,batch_size,step_size)
#loss = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Input Data
X=inpData
Y=outData
#X=np.random.rand(100,inputDim)
#Y=np.random.rand(100,outputDim)

# Resizing
X=X.reshape(totalBatches,step_size,batch_size,inputDim)
Y=Y.reshape(totalBatches,step_size,batch_size,outputDim)

for epoch in range(10):
    for curBatch in range(totalElements/(step_size*batch_size)):
        dataInput=torch.Tensor(X[curBatch])
        #dataInput=[torch.Tensor(x) for x in dataInput]
        
        dataY=torch.Tensor(Y[curBatch])
        #dataY=[torch.Tensor(x) for x in dataInput]
        
        dataOutput=model(dataInput)
        loss=lossCalc(dataOutput,dataY.view(-1,outputDim))
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
    if(epoch % 100==0):
        print("For epoch {}, the loss is {}".format(epoch,loss))

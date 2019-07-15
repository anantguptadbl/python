####################################
# CORRELATION WITH LS
####################################
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class LSTMSimple(nn.Module):
    def __init__(self,inputDim,hiddenDim,batchSize,outputDim):
        super(LSTMSimple,self).__init__()
        torch.manual_seed(1)
        self.lstm=nn.LSTM(inputDim,hiddenDim,1)
        # Hidden state is a tuple of two states, so we will have to initialize two tuples
        self.state_h = torch.randn(1,batchSize,hiddenDim)
        self.state_c = torch.rand(1,batchSize,hiddenDim)
        self.linearModel=nn.Linear(hiddenDim,outputDim)
        
    def forward(self,inputs):
        # LSTM
        output, self.hidden = self.lstm(inputs, (self.state_h,self.state_c) )
        self.state_h=self.state_h.detach()
        self.state_c=self.state_c.detach()
        # LINEAR MODEL
        output=self.linearModel(output)
        return output

def lossCalc(x,y):
    return torch.sum(torch.add(x,-y))
    
# Model Object
batchSize=5
inputDim=10
outputDim=1
stepSize=24
hiddenDim=3
model=LSTMSimple(inputDim,hiddenDim,batchSize,outputDim)
loss = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Input Data
dataInput = torch.randn(stepSize,batchSize,inputDim) 
dataY = torch.randn(stepSize,batchSize,outputDim) 
for epoch in range(10000):
    optimizer.zero_grad()
    dataOutput=model(dataInput)
    curLoss=loss(dataOutput.view(batchSize*stepSize,outputDim),dataY.view(batchSize*stepSize,outputDim))
    curLoss.backward()
    optimizer.step()
    if(epoch % 1000==0):
        print("For epoch {}, the loss is {}".format(epoch,curLoss))

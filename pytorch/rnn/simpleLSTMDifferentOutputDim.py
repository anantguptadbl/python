import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class LSTMSimple(nn.Module):
    def __init__(self,inputDim,hiddenDim,outputDim):
        super(LSTMSimple,self).__init__()
        torch.manual_seed(1)
        self.lstm=nn.LSTM(inputDim,hiddenDim)
        # Hidden state is a tuple of two states, so we will have to initialize two tuples
        self.hidden = (torch.randn(1,1,3) , torch.rand(1,1,3))
        self.linearModel=nn.Linear(hiddenDim,outputDim)
        
    def forward(self,inputs):
        self.fullDataOutput=[]
        for i in inputs:
            self.out,self.hidden = lstm(i.view(1,1,-1),self.hidden)
            # We will now have a linear layer that will convert this to the output size
            self.outLinear=self.linearModel(self.out)
            self.fullDataOutput.append(self.outLinear)
        return self.fullDataOutput

def lossCalc(x,y):
    return torch.sum(torch.add(x,-y)).pow(2) 
    #logsoftmax = nn.LogSoftmax()
    #return torch.mean(torch.sum(- pred  * logsoftmax(soft_targets), 1))
    
# Model Object
inputDim=3
hiddenDim=3
outputDim=2
model=LSTMSimple(inputDim,hiddenDim,outputDim)
#loss = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Input Data
dataInput = [torch.randn(1, 3) for _ in range(5)]  # make a sequence of length 5
dataY = [torch.randn(1, 2) for _ in range(5)]  # make a sequence of length 5
for epoch in range(1000):
    dataOutput=model(dataInput)
    loss=lossCalc(torch.stack(dataOutput).view(5,outputDim),torch.stack(dataY).view(5,outputDim))
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()
    if(epoch % 100==0):
        print("For epoch {}, the loss is {}".format(epoch,loss))

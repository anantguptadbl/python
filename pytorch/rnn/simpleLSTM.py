import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class LSTMSimple(nn.Module):
    def __init__(self,inputDim,hiddenDim):
        super(LSTMSimple,self).__init__()
        torch.manual_seed(1)
        self.lstm=nn.LSTM(inputDim,hiddenDim)
        # Hidden state is a tuple of two states, so we will have to initialize two tuples
        self.hidden = (torch.randn(1,1,3) , torch.rand(1,1,3))
        
    def forward(self,inputs):
        self.fullDataOutput=[]
        for i in inputs:
            self.out,self.hidden = lstm(i.view(1,1,-1),self.hidden)
            self.fullDataOutput.append(self.out)
        return self.fullDataOutput

def lossCalc(x,y):
    return torch.sum(torch.add(x,-y))
    #logsoftmax = nn.LogSoftmax()
    #return torch.mean(torch.sum(- pred  * logsoftmax(soft_targets), 1))
    
# Model Object
model=LSTMSimple(3,3)
loss = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Input Data
dataInput = [torch.randn(1, 3) for _ in range(5)]  # make a sequence of length 5
dataY = [torch.randn(1, 3) for _ in range(5)]  # make a sequence of length 5
for epoch in range(1000):
    dataOutput=model(dataInput)
    loss=lossCalc(torch.stack(dataOutput).view(5,3),torch.stack(dataY).view(5,3))
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()
    if(epoch % 100==0):
        print("For epoch {}, the loss is {}".format(epoch,loss))

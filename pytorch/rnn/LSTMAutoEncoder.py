import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class LSTMAutoEncoder(nn.Module):
    def __init__(self,inputDim,hiddenDim,outputDim,batch_size,step_size):
        super(LSTMAutoEncoder,self).__init__()
        self.inputDim=inputDim
        self.hiddenDim=hiddenDim
        self.outputDim=outputDim
        self.batch_size=batch_size
        self.step_size=step_size
        torch.manual_seed(1)
        # input_size – The number of expected features in the input x
        # hidden_size – The number of features in the hidden state h
        # num_layers – Number of recurrent layers
        self.lstm1=nn.LSTM(self.inputDim,self.hiddenDim)
        self.lstm2=nn.LSTM(self.hiddenDim,self.hiddenDim)
        # Hidden state is a tuple of two states, so we will have to initialize two tuples
        # h_0 of shape (num_layers * num_directions, batch, hidden_size)
        # c_0 of shape (num_layers * num_directions, batch, hidden_size)
        self.hidden1_1 = torch.autograd.variable(torch.randn(1,self.batch_size,self.hiddenDim))
        self.hidden1_2 = torch.autograd.variable(torch.rand(1,self.batch_size,self.hiddenDim))
        self.hidden2_1 = torch.autograd.variable(torch.randn(1,self.batch_size,self.hiddenDim))
        self.hidden2_2 = torch.autograd.variable(torch.randn(1,self.batch_size,self.hiddenDim))
        #self.hidden1 = (torch.randn(1,self.batch_size,self.hiddenDim) , torch.rand(1,self.batch_size,self.hiddenDim))
        #self.hidden2 = (torch.randn(1,self.batch_size,self.hiddenDim) , torch.rand(1,self.batch_size,self.hiddenDim))
        self.linearModel=nn.Linear(self.hiddenDim,self.outputDim)

    def forward(self,inputs):
        self.fullDataOutput=[]
        # STEP 1 : LSTM 1
        # input of shape (seq_len, batch, input_size)
        # h_0 of shape (num_layers * num_directions, batch, hidden_size)
        # c_0 of shape (num_layers * num_directions, batch, hidden_size)
        self.out1,self.hidden1 = self.lstm1(inputs,(self.hidden1_1,self.hidden1_2))
        # STEP 2 : CREATE SEQUENCE FROM THE HIDDEN VECTOR STATE
        self.input2=self.hidden1[0].repeat(self.step_size,1,1)
        # STEP 3 : LSTM 2
        self.out2,self.hidden2 = self.lstm2(self.input2,(self.hidden2_1,self.hidden2_2))
        # STEP 4 : LINEAR
        self.outLinear=self.linearModel(self.out2)
        self.fullDataOutput.append(self.outLinear)
        return torch.stack(self.fullDataOutput).view(-1,self.outputDim)
        
def lossCalc(x,y):
    return torch.sum(torch.add(x,-y)).pow(2) 

# Model Object
inputDim=5
hiddenDim=5
outputDim=5
epochRange=50000

# LSTM Configuration
totalElements=100  # This denotes the number of rows. Each row consits of the number of inputElements
batch_size=5       # This is the number of rows that will be used for gradient update
step_size=5         # This is the number of LSTM cells
totalBatches=totalElements/(step_size*batch_size)


model=LSTMAutoEncoder(inputDim,hiddenDim,outputDim,batch_size,step_size)
#loss = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(),lr=0.0001,amsgrad=True,weight_decay=0.99)

# Input Data
X=np.random.rand(totalElements,inputDim)

# Resizing
X=X.reshape(totalBatches,step_size,batch_size,inputDim)

for epoch in range(epochRange):
    lossVal=0
    for curBatch in range(totalElements/(step_size*batch_size)):
        model.zero_grad()
        dataInput=torch.autograd.Variable(torch.Tensor(X[curBatch]))
        dataOutput=model(dataInput)
        loss=lossCalc(dataOutput,dataInput.view(step_size*batch_size,-1))
        loss.backward()
        lossVal = lossVal + loss
        optimizer.step()
    if(epoch % 1000==0):
        print("For epoch {}, the loss is {}".format(epoch,lossVal))
print("Autoencoder Training completed")

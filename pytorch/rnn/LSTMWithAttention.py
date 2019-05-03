import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class LSTMAutoEncoderWithAttention(nn.Module):
    def __init__(self,inputDim,hiddenDim,outputDim,batch_size,step_size):
        super(LSTMAutoEncoderWithAttention,self).__init__()
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
        self.linearModel=nn.Linear(self.hiddenDim,self.outputDim)
        # ATTENTION WEIGHTS
        self.attentionWeights=torch.autograd.variable(torch.randn(self.step_size,self.step_size))

    def forward(self,inputs):
        self.fullDataOutput=[]
        # STEP 1 : ENCODER LSTM
        # input of shape (seq_len, batch, input_size)
        # h_0 of shape (num_layers * num_directions, batch, hidden_size)
        # c_0 of shape (num_layers * num_directions, batch, hidden_size)
        self.hidden1List=[]
        for x in inputs:
            self.out1,(self.hidden1_1,self.hidden1_2) = self.lstm1(x.view(1,self.batch_size,self.inputDim),(self.hidden1_1,self.hidden1_2))
            self.hidden1List.append(self.hidden1_1)
        
        # DETACHING THE HIDDEN STATE AND CELL STATE for ENCODER LSTM
        self.hidden1_1=self.hidden1_1.detach()
        self.hidden1_2=self.hidden1_2.detach()
        
        # ATTENTION WEIGHTS FOR EACH STEP
        self.attentionOutput=[]
        for x in range(self.attentionWeights.size()[0]):
            self.stepAttentionOutput=torch.zeros(self.batch_size,self.hiddenDim)
            for y in range(self.attentionWeights.size()[1]):
                self.stepAttentionOutput = torch.add(self.stepAttentionOutput , self.hidden1List[y] * torch.autograd.Variable(self.attentionWeights[x][y]))
            self.attentionOutput.append(self.stepAttentionOutput)
            
        # STEP 3 : DECODER LSTM
        self.lstmout2=[]
        for x in self.attentionOutput:
            self.out2,(self.hidden2_1,self.hidden2_2) = self.lstm2(x.view(1,self.batch_size,self.hiddenDim),(self.hidden2_1,self.hidden2_2))
            self.lstmout2.append(self.out2)
        
        # DETACHING THE HIDDEN STATE AND CELL STATE for DECODER LSTM
        self.hidden2_1=self.hidden2_1.detach()
        self.hidden2_2=self.hidden2_2.detach()
        
        
        # STEP 4 : LINEAR
        self.lstmout2=torch.stack(self.lstmout2)
        self.outLinear=self.linearModel(self.lstmout2)
        self.fullDataOutput.append(self.outLinear)
        return torch.stack(self.fullDataOutput).view(-1,self.outputDim)
        
def lossCalc(x,y):
    return torch.sum(torch.add(x,-y)).pow(2) 

# Model Object
inputDim=5
hiddenDim=5
outputDim=5
epochRange=5000

# LSTM Configuration
totalElements=100  # This denotes the number of rows. Each row consits of the number of inputElements
batch_size=5       # This is the number of rows that will be used for gradient update
step_size=5         # This is the number of LSTM cells
totalBatches=totalElements/(step_size*batch_size)


model=LSTMAutoEncoderWithAttention(inputDim,hiddenDim,outputDim,batch_size,step_size)
#loss = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(),lr=0.0001,amsgrad=True,weight_decay=0.85)

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
    if(epoch % 100==0):
        print("For epoch {}, the loss is {}".format(epoch,lossVal))
print("Autoencoder Training completed")

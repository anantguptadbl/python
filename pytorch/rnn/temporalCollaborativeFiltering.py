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
        self.hidden1_2 = torch.autograd.variable(torch.randn(1,self.batch_size,self.hiddenDim))
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
        
        # NORMALIZE THE ATTENTION WEIGHTS
        self.attentionWeightsNormalized = nn.functional.softmax(self.attentionWeights)
        #output = nn.functional.linear(x, w_normalized)
        
        # ATTENTION WEIGHTS FOR EACH STEP
        self.attentionOutput=[]
        for x in range(self.attentionWeightsNormalized.size()[0]):
            self.stepAttentionOutput=torch.zeros(self.batch_size,self.hiddenDim)
            for y in range(self.attentionWeightsNormalized.size()[1]):
                self.stepAttentionOutput = torch.add(self.stepAttentionOutput , self.hidden1List[y] * torch.autograd.Variable(self.attentionWeightsNormalized[x][y]))
            self.attentionOutput.append(self.stepAttentionOutput)
          
        # STEP 3 : DECODER LSTM
        #self.lstmout2=[]
        #for x in self.attentionOutput:
        #    self.out2,(self.hidden2_1,self.hidden2_2) = self.lstm2(x.view(1,self.batch_size,self.hiddenDim),(self.hidden2_1,self.hidden2_2))
        #    self.lstmout2.append(self.out2)
        
        # DETACHING THE HIDDEN STATE AND CELL STATE for DECODER LSTM
        #self.hidden2_1=self.hidden2_1.detach()
        #self.hidden2_2=self.hidden2_2.detach()
        
        
        # STEP 4 : LINEAR
        self.lstmout2=torch.stack(self.attentionOutput)
        self.outLinear=self.linearModel(self.lstmout2)
        self.fullDataOutput.append(self.outLinear)
        return torch.stack(self.fullDataOutput).view(-1,self.outputDim)
      
    def predict(self,inputs):
        hidden1List=[]
        hidden1_1 = torch.autograd.variable(torch.randn(1,1,hiddenDim))
        hidden1_2 = torch.autograd.variable(torch.randn(1,1,hiddenDim))
        for x in inputs:
            out1,(hidden1_1,hidden1_2) = self.lstm1(x.view(1,1,self.inputDim),(hidden1_1,hidden1_2))
            hidden1List.append(hidden1_1)
            
        attentionOutput=[]
        for x in range(self.attentionWeightsNormalized.size()[0]):
            stepAttentionOutput=torch.zeros(1,self.hiddenDim)
            for y in range(self.attentionWeightsNormalized.size()[1]):
                stepAttentionOutput = torch.add(stepAttentionOutput , hidden1List[y] * torch.autograd.Variable(self.attentionWeightsNormalized[x][y]))
            attentionOutput.append(stepAttentionOutput)
        
        # STEP 4 : LINEAR
        lstmout2=torch.stack(attentionOutput)
        outLinear=self.linearModel(lstmout2)
        return outLinear
        #fullDataOutput.append(outLinear)
        #print(len(fullDataOutput))
        #print(fullDataOutput[0].size())
        #return torch.stack(fullDataOutput).view(-1,self.outputDim)  
        
        
def lossCalc(x,y):
    return torch.sum(torch.add(x,-y)).pow(2) 

# Model Object
inputDim=100
hiddenDim=300
outputDim=100
epochRange=20

# LSTM Configuration
#totalElements=100  # This denotes the number of rows. Each row consits of the number of inputElements
batch_size=5       # This is the number of rows that will be used for gradient update
step_size=5         # This is the number of LSTM cells
totalBatches=1000/batch_size


model=LSTMAutoEncoderWithAttention(inputDim,hiddenDim,outputDim,batch_size,step_size)
#loss = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(),lr=0.0001,amsgrad=True,weight_decay=0.8)

# CONCOCTING RANDOM DATA
X=np.random.randint(10,size=(1000,6,100))
X=X-5

for epoch in range(epochRange):
    lossVal=0
    for curBatch in range(totalBatches):
        model.zero_grad()
        dataInput=torch.autograd.Variable(torch.Tensor(X[batch_size*curBatch:batch_size*(curBatch+1),0:step_size,:]))
        dataOutput=model(dataInput)
        loss=lossCalc(dataOutput,torch.autograd.Variable(torch.Tensor(X[batch_size*curBatch:batch_size*(curBatch+1),1:step_size+1,:])).view(batch_size*step_size,inputDim))
        loss.backward()
        lossVal = lossVal + loss
        optimizer.step()
    if(epoch % 10==0):
        print("For epoch {}, the loss is {}".format(epoch,lossVal))
print("Autoencoder Training completed")

# PREDICTION
print("Prediction Started")
a=np.random.randint(1,size=(5,100)).astype(np.float)
for i in range(5):
    a[i][i]=5
print(type(a))
prediction=model.predict(torch.tensor(a).float())
print("Prediction Completed")

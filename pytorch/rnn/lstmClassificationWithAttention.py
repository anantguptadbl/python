#############################################################
# COMBINING MULTIPLE TIMESTEPS FOR A SINGLE CLASSIFICATION
############################################################
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy

class AttLayer(nn.Module):
    def __init__(self, hiddenDim,attentionDim):
        #self.init = initializers.get('normal')
        #self.supports_masking = True
        #self.attention_dim = attention_dim
        super(AttLayer, self).__init__()
        self.hiddenDim=hiddenDim
        self.attentionDim=attentionDim
        self.W = torch.randn(hiddenDim, attentionDim)
        self.b = torch.randn(self.attentionDim,)
        self.u = torch.randn(self.attentionDim, 1)
        self.trainable_weights = [self.W, self.b, self.u]
        self.tanh=torch.nn.Tanh()


    def forward(self, x):
        # size of x :[batch_size, sel_len, attention_dim]
        # size of u :[batch_size, attention_dim]
        # uit = tanh(xW+b)
        uit = self.tanh(torch.add(torch.matmul(x, self.W), self.b))
        ait = torch.matmul(uit, self.u)
        ait = torch.squeeze(ait, -1)
        ait = torch.exp(ait)
        #ait = ait / torch.sum(ait, dim=1,keepdim=True)
        ait = ait / torch.sum(ait, dim=0,keepdim=True)
        #print("After sum divide the size of ait is {0}".format(ait.size()))
        #ait = K.expand_dims(ait)
        dimValues=tuple(ait.size()) + (1,)
        self.ait=ait.view(dimValues)
        weighted_input = x * self.ait
        output = torch.sum(weighted_input, dim=0)
        return output

class LSTMSimple(nn.Module):
    def __init__(self,inputDim,hiddenDim,batchSize,outputDim,attentionDim):
        # Constructor
        super(LSTMSimple,self).__init__()
        # Seed Value
        torch.manual_seed(1)
        
        # Vaiable Initialization
        self.inputDim=inputDim
        self.hiddenDim=hiddenDim
        self.batchSize=batchSize
        self.outputDim=outputDim
        self.attentionDim=attentionDim
        
        # Model Initialization
        self.lstm=nn.LSTM(inputDim,hiddenDim,1)
        
        # Hidden state and Cell State Initialization
        self.state_h = torch.randn(1,batchSize,hiddenDim)
        self.state_c = torch.rand(1,batchSize,hiddenDim)
        
        # Add the Attention Layer
        self.attLayer=AttLayer(hiddenDim,attentionDim)
        
        # Final Linear Model Initialization
        self.linearModel=nn.Linear(hiddenDim,outputDim)
        #self.linearModel=LinearModel1(attentionDim,outputDim)
        
    def forward(self,inputs):
        # LSTM
        hiddenLayers=[]
        outputs=[]
        for curInput in inputs:
            output, (self.state_h,self.state_c) = self.lstm(curInput.view(1,batchSize,inputDim), (self.state_h,self.state_c) )
            self.state_h=self.state_h.detach()
            self.state_c=self.state_c.detach()
            hiddenLayers.append(copy.copy(self.state_h))
            outputs.append(copy.copy(output))
        
        # Stacking the data
        output=torch.stack(outputs).view(-1,self.batchSize,self.hiddenDim)
        
        # Attention Mechanism
        attOutput=self.attLayer(output)
        # LINEAR MODEL
        output=self.linearModel(attOutput)
        return output

def lossCalc(x,y):
    return torch.sum(torch.add(x,-y))
    
# Model Object
batchSize=5
inputDim=10
outputDim=1
stepSize=24
hiddenDim=3
attentionDim=5
model=LSTMSimple(inputDim,hiddenDim,batchSize,outputDim,attentionDim)
loss = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Input Data
dataInput = torch.randn(stepSize,batchSize,inputDim) 
#dataY = torch.randn(stepSize,batchSize,outputDim) 
dataY = torch.randn(batchSize,outputDim) 
for epoch in range(1000):
    optimizer.zero_grad()
    dataOutput=model(dataInput)
    #curLoss=loss(dataOutput.view(batchSize*stepSize,outputDim),dataY.view(batchSize*stepSize,outputDim))
    curLoss=loss(dataOutput.view(batchSize,outputDim),dataY.view(batchSize,outputDim))
    curLoss.backward()
    optimizer.step()
    if(epoch % 100==0):
        print("For epoch {}, the loss is {}".format(epoch,curLoss))
    
# We will now find out which feature is the most important for a particular input row
weights=model.attLayer.ait.detach().data.numpy()
# For the first row
maxAffectingTimeStep=np.argmax(weights[:,0,:])

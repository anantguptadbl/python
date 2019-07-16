import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy

class AttLayer1(nn.Module):
    def __init__(self, hiddenDim,attentionDim):
        #self.init = initializers.get('normal')
        #self.supports_masking = True
        #self.attention_dim = attention_dim
        super(AttLayer1, self).__init__()
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
        ait = ait / torch.sum(ait, dim=2,keepdim=True)
        #print("After sum divide the size of ait is {0}".format(ait.size()))
        #ait = K.expand_dims(ait)
        dimValues=tuple(ait.size()) + (1,)
        self.ait=ait.view(dimValues)
        weighted_input = x * self.ait
        output = torch.sum(weighted_input, dim=2)
        return output

class AttLayer2(nn.Module):
    def __init__(self, hiddenDim,attentionDim):
        #self.init = initializers.get('normal')
        #self.supports_masking = True
        #self.attention_dim = attention_dim
        super(AttLayer2, self).__init__()
        self.hiddenDim=hiddenDim
        self.attentionDim=attentionDim
        self.W = torch.randn(self.hiddenDim, attentionDim)
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
    def __init__(self,inputDim,hiddenDim,batchSize,outputDim,attentionDim1,attentionDim2):
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
        self.lstm1=nn.LSTM(inputDim,hiddenDim,1)
        self.lstm2=nn.LSTM(1,hiddenDim,1)
        
        # Hidden state and Cell State Initialization
        self.state_h = torch.randn(1,batchSize,hiddenDim)
        self.state_c = torch.rand(1,batchSize,hiddenDim)
        
        # Add the Attention Layer for the variable
        self.attLayer1=AttLayer1(1,attentionDim1)
        
        # Add the Attention Layer for the timestep
        self.attLayer2=AttLayer2(hiddenDim,attentionDim2)
        
        # Final Linear Model Initialization
        self.linearModel=nn.Linear(hiddenDim,outputDim)
        #self.linearModel=LinearModel1(attentionDim,outputDim)
        
    def forward(self,inputs):
        # LSTM 1 : This will apply attn on the input 100 variables
        #layer1HiddenLayers=[]
        #layer1Outputs=[]
        #for curInput in inputs:
        #    output, (self.state_h,self.state_c) = self.lstm1(curInput.view(1,batchSize,inputDim), (self.state_h,self.state_c) )
        #    self.state_h=self.state_h.detach()
        #    self.state_c=self.state_c.detach()
        #    layer1HiddenLayers.append(copy.copy(self.state_h))
        #    layer1Outputs.append(copy.copy(output))
        
        # Stacking Layer1 Data
        #layer1Output=torch.stack(layer1Outputs).view(-1,self.batchSize,self.hiddenDim)
        
        # Layer 1 Attn
        #layer1AttOutput=self.attLayer1(layer1Output)
        layer1AttOutput=self.attLayer1(inputs.view(-1,self.batchSize,self.inputDim,1))
        
        # LSTM 2
        hiddenLayers=[]
        outputs=[]
        for curInput in layer1AttOutput:
            output, (self.state_h,self.state_c) = self.lstm2(curInput.view(1,batchSize,1), (self.state_h,self.state_c) )
            self.state_h=self.state_h.detach()
            self.state_c=self.state_c.detach()
            hiddenLayers.append(copy.copy(self.state_h))
            outputs.append(copy.copy(output))
        
        # Stacking the data
        output=torch.stack(outputs).view(-1,self.batchSize,self.hiddenDim)
        # Attention Mechanism
        attOutput=self.attLayer2(output)
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
attentionDim1=5
attentionDim2=4
model=LSTMSimple(inputDim,hiddenDim,batchSize,outputDim,attentionDim1,attentionDim2)
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
        
# We will now find out which timestep is the most important
weights=model.attLayer2.ait.detach().data.numpy()
# For the first row
weights[:,0,:]

# We will now find out which feature was the most important
featureWeights=model.attLayer1.ait.detach().data.numpy()
featureWeights[0,0,:,:]

# Read the data
import numpy as np

numMovies=9724
fullData=np.load("fullData.npy")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import time

torch.manual_seed(42)


class AttLayer(nn.Module):
    def __init__(self, attention_dim, inputDim,batchSize):
        super(AttLayer, self).__init__()
        self.attention_dim = attention_dim
        self.inputDim=inputDim
        self.batchSize=batchSize
        self.W = Variable(torch.randn(self.inputDim, self.attention_dim), name='W').cuda()
        self.b = Variable(torch.randn(self.attention_dim, ), name='b').cuda()
        self.u = Variable(torch.randn(self.attention_dim,1), name='u').cuda()
        self.trainable_weights = [self.W, self.b, self.u]

    def forward(self, x):
        # size of x :[batch_size, sel_len, attention_dim]
        # size of u :[batch_size, attention_dim]
        # uit = tanh(xW+b)
        x=x.view(self.batchSize,inputDim).cuda()
        uit= torch.tanh(torch.add(torch.matmul(x,self.W),self.b)).cuda()
        ait=torch.matmul(uit,self.u).cuda()
        # Squeeze
        ait=torch.exp(ait).cuda()
        ait=ait / ((torch.sum(ait,axis=1)).view(self.batchSize,1))
        self.weighted_input=x*ait
        output = torch.sum(self.weighted_input, axis=1).cuda()
        #return(self.weighted_input)
        return(output)
    
class AttLayer2(nn.Module):
    def __init__(self, attention_dim, inputDim,batchSize,stepSize):
        #self.init = initializers.get('normal')
        #self.supports_masking = True
        super(AttLayer2, self).__init__()
        self.attention_dim = attention_dim
        self.inputDim=inputDim
        self.batchSize=batchSize
        self.stepSize=stepSize
        self.W = Variable(torch.randn(self.inputDim, self.attention_dim), name='W').cuda()
        self.b = Variable(torch.randn(self.attention_dim, ), name='b').cuda()
        self.u = Variable(torch.randn(self.attention_dim,1), name='u').cuda()
        self.trainable_weights = [self.W, self.b, self.u]

    def forward(self, x):
        #print("In att layer 2 x = {0}".format(x.size()))
        x=x.view(self.batchSize,self.stepSize,self.inputDim).cuda()
        #print("In att layer the size is {0}".format(x.size()))
        uit= torch.tanh(torch.add(torch.matmul(x,self.W),self.b)).cuda()
        ait=torch.matmul(uit,self.u).cuda()
        ait=torch.exp(ait).cuda()
        #print("AIT size is {0}".format(ait.size()))
        #print("AIT sum size is {0}".format((torch.sum(ait,axis=1)).size()))
        ait=ait / ((torch.sum(ait,axis=1)).view(self.batchSize,1,1))
        self.weighted_input=x*ait
        output = torch.sum(self.weighted_input, axis=1).cuda()
        return(output)

class singleNeuralNetwork(nn.Module):
    def __init__(self,inputDim,outputDim,k,T1,T2):
        super(singleNeuralNetwork,self).__init__()
        self.inputDim=inputDim
        self.outputDim=outputDim
        self.k=k
        self.w=Variable(torch.randn(k,inputDim,outputDim)).cuda()
        self.beta=Variable(torch.randn(k)).cuda()
        self.T1=T1
        self.T2=T2
        self.mu=Variable(torch.randn(outputDim)).cuda()
        
    def forward(self,x):
        self.w=torch.clamp(self.w,self.T1).cuda()
        self.beta=torch.clamp(self.beta,self.T2).cuda()
        results=torch.zeros(x.size()[0],self.outputDim).cuda()
        for curK in range(self.k):
            results= torch.add(results,(torch.matmul(x,self.w[curK]) * self.beta[curK])).cuda()
        return(results)
        
    
class autoencoder(nn.Module):
    def __init__(self,inputDim,linearOutputDim):
        super(autoencoder, self).__init__()
        self.inputDim=inputDim
        self.firstLayer=singleNeuralNetwork(inputDim,linearOutputDim,10,0.9,0.9)
        #self.encoder = nn.Sequential(
        #    nn.Linear(inputDim,100).cuda(),
        #    nn.ReLU(True),
        #    nn.Linear(100,50).cuda(),
        #    nn.ReLU(True)).cuda()
        #self.decoder = nn.Sequential(
        #    nn.Linear(50, 100).cuda(),
        #    nn.ReLU(True),
        #    nn.Linear(100,inputDim).cuda(),
        #    nn.ReLU(True)
        #    ).cuda()
    def forward(self, x):
        x=self.firstLayer(x)
        #x = self.encoder(x).cuda()
        #x = self.decoder(x).cuda()
        return x
        

class Model(nn.Module):
    def __init__(self,inputDim,linearOutputDim,attentionDim,hiddenDim,batchSize,stepSize):
        super(Model,self).__init__()
        torch.manual_seed(1)
        self.stepSize=stepSize
        self.hiddenDim=hiddenDim
        self.inputDim=inputDim
        self.batchSize=batchSize
        self.linearOutputDim=linearOutputDim
        hiddenDim2=50
        self.encoderDim=50
        # Init AutoEncoder
        self.autoencoderLayer=autoencoder(self.inputDim,linearOutputDim)
        # Att Layer
        #self.attLayer1=AttLayer(50,self.inputDim,self.batchSize*self.stepSize).cuda()
        self.attLayer2=AttLayer2(attentionDim,self.hiddenDim,self.batchSize,self.stepSize).cuda()
        # LSTM Encoder
        self.lstm1=nn.LSTM(self.encoderDim,hiddenDim,1).cuda()
        self.lstm2=nn.LSTM(hiddenDim,hiddenDim2,1).cuda()
        self.state_h1 = torch.randn(1,batchSize,hiddenDim).cuda()
        self.state_c1 = torch.rand(1,batchSize,hiddenDim).cuda()
        self.hidden1_1 = Variable(torch.randn(1,self.batchSize,self.hiddenDim)).cuda()
        self.hidden1_2 = Variable(torch.randn(1,self.batchSize,self.hiddenDim)).cuda()
        # LSTM Decoder
        self.state_h2 = torch.randn(1,batchSize,hiddenDim2).cuda()
        self.state_c2 = torch.rand(1,batchSize,hiddenDim2).cuda()
        # Linear Model
        self.linearModel=nn.Linear(hiddenDim2,inputDim*5).cuda()
        # SoftMax
        self.smax=nn.Softmax()
        
    def forward(self,inputs):
        # ATT 1
        #inputs=self.attLayer1(inputs.view(-1,self.inputDim)).view(self.stepSize,self.batchSize,1).cuda()
        # AutoEncoder
        #aeoutput=self.autoencoderLayer(inputs.view(-1,self.inputDim))
        #inputs=self.autoencoderLayer.encoder(inputs.view(-1,self.inputDim)).view(self.stepSize,self.batchSize,self.encoderDim)
        inputs=self.autoencoderLayer(inputs.view(-1,self.inputDim)).view(self.stepSize,self.batchSize,self.linearOutputDim)
        # Then we will do the encoding
        #print("Output of inputs from the first layer is {0}".format(inputs.size()))
        # LSTM 1
        lstmState=[]
        for x in inputs:
            self.out2,(self.hidden1_1,self.hidden1_2) = self.lstm1(x.view(1,self.batchSize,self.linearOutputDim),(self.hidden1_1,self.hidden1_2))
            lstmState.append(self.hidden1_1)
        
        self.hidden1_1=self.hidden1_1.detach()
        self.hidden1_2=self.hidden1_2.detach()
        
        # We will now run Attention on the hidden states
        inputNew=torch.stack(lstmState).view(self.stepSize,self.batchSize,self.hiddenDim).cuda()
        #print("Shape of inputNew is {0}".format(inputNew.size()))
        self.inputNew=self.attLayer2(inputNew)
        
        
        #output, self.hidden1 = self.lstm1(inputs, (self.state_h1,self.state_c1) )
        #inputNew = torch.cat(self.stepSize*[self.hidden1[1]]).reshape(self.stepSize,self.batchSize,-1)
        # LSTM 2
        output, self.hidden2 = self.lstm2(inputNew, (self.state_h2,self.state_c2) )
        #self.state_h=self.state_h.detach()
        #self.state_c=self.state_c.detach()
        # LINEAR MODEL
        output=self.linearModel(output).cuda()
        output=self.smax(output.view(-1,5)).cuda()
        return output
    
# Model Object
batchSize=32
inputDim=numMovies
linearOutputDim=50
stepSize=10
hiddenDim=100
attentionDim=50
model=Model(inputDim,linearOutputDim,attentionDim,hiddenDim,batchSize,stepSize).cuda()
#loss = torch.nn.MSELoss()
loss = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001,weight_decay=0.00001)

# Input Data
batches=int(fullData.shape[0] / batchSize)

startTime=time.time()
for epoch in range(100000):
    totalLoss=0
    for curBatch in range(batches):
        inpData=Variable(torch.from_numpy(fullData[curBatch*batchSize:(curBatch+1)*batchSize].reshape(stepSize,batchSize,numMovies,1).astype(np.float32))).cuda()
        optimizer.zero_grad()
        dataOutput=model(inpData).cuda()
        curLoss=loss(dataOutput,inpData.view(-1).type(torch.LongTensor).cuda())
        curLoss.backward()
        optimizer.step()
        totalLoss=totalLoss + curLoss.item()
    if(epoch % 10==0):
        print("For epoch {0}, the loss is {1} in time {2}".format(epoch,totalLoss,(time.time()-startTime)/60 ))

model.zero_grad()
dataOutput=model(inpData)
r1=model.attLayer2
attLayer2=r1.weighted_input.cpu().detach().numpy()


# Explanability
# Item
impItem1=np.where(np.argsort(np.sum(np.sum(model.autoencoderLayer.firstLayer.w.cpu().detach().numpy(),axis=0),axis=1))==0)[0][0]
impItem2=np.where(np.argsort(np.sum(np.sum(model.autoencoderLayer.firstLayer.w.cpu().detach().numpy(),axis=1),axis=1))==0)[0][0]
impItem3=np.where(np.argsort(np.sum(np.sum(model.autoencoderLayer.firstLayer.w.cpu().detach().numpy(),axis=2),axis=1))==0)[0][0]

# Time Step
impTimeSteps1=np.where(np.argsort(np.sum(attLayer2,axis=2),axis=1)==0)[1]
impTimeSteps2=np.where(np.argsort(np.sum(attLayer2,axis=2),axis=1)==1)[1]
impTimeSteps3=np.where(np.argsort(np.sum(attLayer2,axis=2),axis=1)==2)[1]

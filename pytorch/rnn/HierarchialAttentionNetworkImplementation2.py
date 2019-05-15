#import pandas as pd


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class LSTMHierarchialAttention2(nn.Module):
    def __init__(self,inputDim,hiddenDim,outputDim,batch_size1,batch_size2,numWords,numSentences):
        super(LSTMHierarchialAttention2,self).__init__()
        self.inputDim=inputDim
        self.hiddenDim=hiddenDim
        self.attentionDim=self.hiddenDim
        self.outputDim=outputDim
        self.batch_size1=batch_size1
        self.batch_size2=batch_size2
        self.step_size1=numWords
        self.step_size2=numSentences
        torch.manual_seed(1)
        self.lstm1=nn.LSTM(self.inputDim,self.hiddenDim)
        self.lstm2=nn.LSTM(self.hiddenDim,self.hiddenDim)
        self.hidden1_1 = torch.autograd.variable(torch.randn(1,self.batch_size1,self.hiddenDim))
        self.hidden1_2 = torch.autograd.variable(torch.randn(1,self.batch_size1,self.hiddenDim))
        self.hidden2_1 = torch.autograd.variable(torch.randn(1,self.batch_size2,self.hiddenDim))
        self.hidden2_2 = torch.autograd.variable(torch.randn(1,self.batch_size2,self.hiddenDim))
        
        # Final Linear Model with Sigmoid
        self.finalLayer=nn.Sequential(
            nn.Linear(self.hiddenDim,1),
            nn.Softmax()
        )
        
        # Attention Level 1
        self.attention1W=torch.autograd.variable(torch.randn(self.hiddenDim,self.attentionDim))
        self.attention1B=torch.autograd.variable(torch.randn(self.attentionDim))
        self.attention1U=torch.autograd.variable(torch.randn(self.attentionDim,1))
        
        # Attention Level 2
        self.attention2W=torch.autograd.variable(torch.randn(self.hiddenDim,self.attentionDim))
        self.attention2B=torch.autograd.variable(torch.randn(self.attentionDim))
        self.attention2U=torch.autograd.variable(torch.randn(self.attentionDim,1))
        
    def forward(self,inputs):
        self.sentenceEncodingHiddenList=[]
        # This is the start of the sentence
        for curSentence in inputs:
            self.wordEncodingHiddenList=[]
            # This is at the word level
            for curWord in curSentence:
                self.out1,(self.hidden1_1,self.hidden1_2) = self.lstm1(curWord.view(1,self.batch_size1,self.inputDim),(self.hidden1_1,self.hidden1_2))
                self.wordEncodingHiddenList.append(self.hidden1_1)
            
            # DETACHING THE HIDDEN STATE AND CELL STATE for ENCODER LSTM
            self.hidden1_1=self.hidden1_1.detach()
            self.hidden1_2=self.hidden1_2.detach()
            self.wordEncodingHiddenList1=torch.stack(self.wordEncodingHiddenList)
            
            
            # ATTENTION IN A DIFFERENT FASHION
            self.attention1uit=torch.tanh(torch.add(torch.matmul(self.wordEncodingHiddenList1,self.attention1W),self.attention1B))
            self.attention1uit1=torch.matmul(self.attention1uit,self.attention1U)
            self.attention1res=torch.exp(torch.squeeze(self.attention1uit1,-1))
            #self.attention1res=torch.sum(self.attention1res)
            self.attention1res1 = self.attention1res / (torch.sum(self.attention1res, dim=1, keepdim=True) + torch.exp(torch.Tensor(1)))
            self.attention1res2=self.attention1res1.view(self.attention1res1.size()[0],1)
            self.weighted_input1 = self.wordEncodingHiddenList1.view(self.step_size1,self.hiddenDim) * self.attention1res2
            self.output1 = torch.sum(self.weighted_input1, dim=0)
            self.sentenceEncodingHiddenList.append(self.output1)

        # We have performed the attention at the first level of the hierarchy
        self.sentenceEncodingHiddenList1=torch.stack(self.sentenceEncodingHiddenList)
        self.rowEncodingList1=[]
        for curSentence in self.sentenceEncodingHiddenList1:
            self.out2,(self.hidden2_1,self.hidden2_2) = self.lstm2(curSentence.view(1,self.batch_size2,self.hiddenDim),(self.hidden2_1,self.hidden2_2))
            self.rowEncodingList1.append(self.hidden2_1) 
            
        # DETACHING THE HIDDEN STATE AND CELL STATE for ENCODER LSTM
        self.hidden2_1=self.hidden2_1.detach()
        self.hidden2_2=self.hidden2_2.detach()
        
        # ATTENTION IN A DIFFERENT FASHION
        self.rowEncodingList=torch.stack(self.rowEncodingList1)
        self.attention2uit=torch.tanh(torch.add(torch.matmul(self.rowEncodingList,self.attention2W),self.attention2B))
        self.attention2uit1=torch.matmul(self.attention2uit,self.attention2U)
        self.attention2res=torch.exp(torch.squeeze(self.attention2uit1,-1))
        #self.attention1res=torch.sum(self.attention1res)
        self.attention2res1 =self.attention2res / (torch.sum(self.attention2res, dim=1, keepdim=True) + torch.exp(torch.Tensor(1)))
        self.attention2res2=self.attention2res1.view(self.attention2res1.size()[0],1)
        self.weighted_input2 = self.rowEncodingList.view(self.step_size2,self.hiddenDim) * self.attention2res2
        self.output2 = torch.sum(self.weighted_input2, dim=0).view(1,self.hiddenDim)
        
        # We will now apply a linear model to convert the 40 dimensions to 2 dimensions for output
        self.finalOutput=self.finalLayer(self.output2)
        return self.finalOutput
    
def lossCalc(x,y):
    return torch.sum(torch.add(x,-y)).pow(2) 

# Model Object
inputDim=10
hiddenDim=20
outputDim=20
epochRange=1000

    
# LSTM Configuration
numRows=50  # This denotes the number of rows. Each row consits of the number of inputElements
numSentences=5       # This is the number of rows that will be used for gradient update
numWords=10         # This is the number of LSTM cells
batch_size1=1
batch_size2=1
totalBatches=int(numRows/batch_size1)

model=LSTMHierarchialAttention2(inputDim,hiddenDim,outputDim,batch_size1,batch_size2,numWords,numSentences)
#loss = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(),lr=0.0001,amsgrad=True,weight_decay=0.85)

# Input Data
X=np.random.rand(numRows * numSentences * numWords,inputDim)

# Resizing
X=X.reshape(totalBatches,numSentences,numWords,inputDim)

# Creating Y
Y=np.random.randint(2,size=(numRows,1))

for epoch in range(epochRange):
    lossVal=0
    for curBatch in range(totalBatches):
        model.zero_grad()
        dataInput=torch.autograd.Variable(torch.Tensor(X[curBatch]))
        dataOutput=model(dataInput)
        loss=lossCalc(dataOutput,torch.Tensor(Y[curBatch]))
        loss.backward()
        lossVal = lossVal + loss
        optimizer.step()
    if(epoch % 1==0):
        print("For epoch {}, the loss is {}".format(epoch,lossVal))
print("Model Training completed")

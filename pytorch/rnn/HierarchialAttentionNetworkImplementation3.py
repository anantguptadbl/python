#import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class LSTMHierarchialAttention3(nn.Module):
    def __init__(self,inputDim,hiddenDim,outputDim,batch_size1,batch_size2,numWords,numSentences):
        super(LSTMHierarchialAttention3,self).__init__()
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
        
        # We will skip Setence Encoding for now
  
        # Final Linear Model with Sigmoid
        self.finalLayer=nn.Sequential(
            nn.Linear(self.hiddenDim,1),
            nn.Sigmoid()
        )
        
        # Attention Level 1 : Word Level
        self.attention1W=torch.autograd.variable(torch.randn(self.step_size2,self.step_size1,self.hiddenDim,self.attentionDim))
        self.attention1B=torch.autograd.variable(torch.randn(self.step_size2,self.step_size1,self.attentionDim))
        self.attention1U=torch.autograd.variable(torch.randn(self.step_size2,self.step_size1,self.attentionDim,1))
        
        # Attention Level 2
        self.attention2W=torch.autograd.variable(torch.randn(self.step_size2,self.hiddenDim,self.attentionDim))
        self.attention2B=torch.autograd.variable(torch.randn(self.step_size2,self.attentionDim))
        self.attention2U=torch.autograd.variable(torch.randn(self.step_size2,self.attentionDim,1))
        
    def forward(self,inputs):
        self.sentenceEncodingHiddenList=[]
        # This is the start of the sentence
        for curSentence in inputs:
            self.wordEncodingHiddenList=[]
            # This is at the word level
            for curWord in curSentence:
                out1,(self.hidden1_1,self.hidden1_2) = self.lstm1(curWord.view(1,self.batch_size1,self.inputDim),(self.hidden1_1,self.hidden1_2))
                self.wordEncodingHiddenList.append(copy.copy(self.hidden1_1))
                    # DETACHING THE HIDDEN STATE AND CELL STATE for ENCODER LSTM
            self.hidden1_1=self.hidden1_1.detach()
            self.hidden1_2=self.hidden1_2.detach()
            self.sentenceEncodingHiddenList.append(copy.copy(torch.stack(self.wordEncodingHiddenList)))
            
        # Once we have got all the hidden states for all the sentences, we will now pass them through the attention network
        self.sentenceWordAttention=[]
        for i,curSentence in enumerate(self.sentenceEncodingHiddenList):
            wordAttention=[]
            for j,curWord in enumerate(curSentence):
                self.attention1uit=torch.tanh(torch.add(torch.matmul(curWord,self.attention1W[i][j]),self.attention1B[i][j]))
                self.attention1uit1=torch.matmul(self.attention1uit,self.attention1U[i][j])
                self.attention1res=torch.exp(torch.squeeze(self.attention1uit1,-1)[0])
                wordAttention.append(copy.copy(self.attention1res))

            self.attention1res0=torch.stack(wordAttention)
            self.attention1res1 = (1.0000000 * self.attention1res0)/ torch.sum(self.attention1res0, dim=0, keepdim=True)
            self.attention1res2=self.attention1res1.view(self.attention1res1.size()[0],1)
            self.weighted_input1 = curSentence.view(self.step_size1,self.hiddenDim) * self.attention1res2
            self.output1 = torch.sum(self.weighted_input1, dim=0)
            self.sentenceWordAttention.append(self.output1)
            
        # We have performed the attention at the word level. Now we will do at the sentence level
        
        # We will skip sentence level encoding for now as given in the paper

        self.rowEncodingList1=[]
        for curSentence in self.sentenceWordAttention:
            #print("Size of curSentence is {0}".format(curSentence.size()))
            self.out2,(self.hidden2_1,self.hidden2_2) = self.lstm2(curSentence.view(1,self.batch_size2,self.hiddenDim),(self.hidden2_1,self.hidden2_2))
            self.rowEncodingList1.append(copy.copy(self.hidden2_1)) 
        
        #print("The size of self.rowEncodingList1 is {0}".format(len(self.rowEncodingList1)))
        # DETACHING THE HIDDEN STATE AND CELL STATE for ENCODER LSTM
        self.hidden2_1=self.hidden2_1.detach()
        self.hidden2_2=self.hidden2_2.detach()
        
        # SENTENCE LEVEL
        self.sentenceAttention=[]
        for i,curSentence in enumerate(self.rowEncodingList1):
            self.attention2uit=torch.tanh(torch.add(torch.matmul(curWord,self.attention2W[i]),self.attention2B[i]))
            self.attention2uit1=torch.matmul(self.attention2uit,self.attention2U[i])
            self.attention2res=torch.exp(torch.squeeze(self.attention2uit1,-1)[0])
            self.sentenceAttention.append(copy.copy(self.attention2res))

        self.attention2res0=torch.stack(self.sentenceAttention)
        self.attention2res1 =(1.000000000 * self.attention2res0) / (torch.sum(self.attention2res0, dim=0, keepdim=True)) 
        self.attention2res2=self.attention2res1.view(self.attention2res1.size()[0],1)
        self.weighted_input2 = torch.stack(self.rowEncodingList1).view(self.step_size2,self.hiddenDim) * self.attention2res2
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
numRows=20         # This denotes the number of rows. Each row consits of the number of inputElements
numSentences=5     # This is the number of rows that will be used for gradient update
numWords=10        # This is the number of LSTM cells
batch_size1=1
batch_size2=1
totalBatches=int(numRows/batch_size1)

model=LSTMHierarchialAttention3(inputDim,hiddenDim,outputDim,batch_size1,batch_size2,numWords,numSentences)
optimizer = optim.Adam(model.parameters(),lr=0.00001,amsgrad=True,weight_decay=0.99)

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
    if(epoch % 10==0):
        print("For epoch {}, the loss is {}".format(epoch,lossVal))
print("Autoencoder Training completed")

# Now we will study the weights that has been trained on the data
print(torch.stack(model.sentenceAttention).detach().numpy())
print(torch.stack(model.sentenceWordAttention).detach().numpy())

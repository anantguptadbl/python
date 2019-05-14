#import pandas as pd


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class LSTMHierarchialAttention(nn.Module):
    def __init__(self,inputDim,hiddenDim,outputDim,batch_size1,batch_size2,numWords,numSentences):
        super(LSTMHierarchialAttention,self).__init__()
        self.inputDim=inputDim
        self.hiddenDim=hiddenDim
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
        
         # ATTENTION WEIGHTS WORD LEVEL
        self.attentionWeightsWords=torch.autograd.variable(torch.randn(1,self.step_size1))
        self.attentionWeightsSentences=torch.autograd.variable(torch.randn(1,self.step_size2))
        
        # Final Linear Model with Sigmoid
        self.finalLayer=nn.Sequential(
            nn.Linear(self.hiddenDim,1),
            nn.Sigmoid()
        )
        
    def forward(self,inputs):
        #print("the size of the inputs is {}".format(inputs.size()))
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
        
            # NORMALIZE THE ATTENTION WEIGHTS
            self.attentionWeightsWordsNormalized = nn.functional.softmax(self.attentionWeightsWords)
            self.wordEncodingHiddenList1=torch.stack(self.wordEncodingHiddenList).view(self.step_size1,self.hiddenDim)
            self.stepAttentionOutput=torch.matmul(self.attentionWeightsWordsNormalized,self.wordEncodingHiddenList1)
			
            # NOW WE WILL USE THIS TO ARRIVE AT A SINGLE VECTOR FOR EACH SENTENCE

            self.sentenceEncodingHiddenList.append(self.stepAttentionOutput)
        
        self.sentenceEncodingHiddenList1=torch.stack(self.sentenceEncodingHiddenList)
        self.rowEncodingList=[]
        for curSentence in self.sentenceEncodingHiddenList1:
            self.out2,(self.hidden2_1,self.hidden2_2) = self.lstm2(curSentence.view(1,self.batch_size2,self.hiddenDim),(self.hidden2_1,self.hidden2_2))
            self.rowEncodingList.append(self.hidden2_1) 
            
        # DETACHING THE HIDDEN STATE AND CELL STATE for ENCODER LSTM
        self.hidden2_1=self.hidden2_1.detach()
        self.hidden2_2=self.hidden2_2.detach()
        
        # NORMALIZE THE ATTENTION WEIGHTS
        self.attentionWeightsSentencesNormalized = nn.functional.softmax(self.attentionWeightsSentences)
        self.rowEncodingList1=torch.stack(self.rowEncodingList).view(self.step_size2,self.hiddenDim)
        self.rowAttentionOutput=torch.matmul(self.attentionWeightsSentencesNormalized,self.rowEncodingList1)
        
        
        # We will now apply a linear model to convert the 40 dimensions to 2 dimensions for output
        self.finalOutput=self.finalLayer(self.rowAttentionOutput)
        
        return self.finalOutput
    
def lossCalc(x,y):
    return torch.sum(torch.add(x,-y)).pow(2) 

# Model Object
inputDim=20
hiddenDim=40
outputDim=40
epochRange=1000

    
# LSTM Configuration
numRows=10  # This denotes the number of rows. Each row consits of the number of inputElements
numSentences=5       # This is the number of rows that will be used for gradient update
numWords=10         # This is the number of LSTM cells
batch_size1=1
batch_size2=1
totalBatches=int(numRows/batch_size1)

model=LSTMHierarchialAttention(inputDim,hiddenDim,outputDim,batch_size1,batch_size2,numWords,numSentences)
#loss = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(),lr=0.0001,amsgrad=True,weight_decay=0.85)

# Input Data
X=np.random.rand(numRows * numSentences * numWords,inputDim)

# Resizing
X=X.reshape(totalBatches,numSentences,numWords,inputDim)

# Creating Y
Y=np.random.randint(2,size=(numRows))

for epoch in range(epochRange):
    lossVal=0
    for curBatch in range(totalBatches):
        model.zero_grad()
        dataInput=torch.autograd.Variable(torch.Tensor(X[curBatch]))
        dataOutput=model(dataInput)
        #print(dataOutput.size())
        loss=lossCalc(dataOutput,dataInput.view(step_size*batch_size1,-1))
        loss.backward()
        lossVal = lossVal + loss
        optimizer.step()
    if(epoch % 1==0):
        print("For epoch {}, the loss is {}".format(epoch,lossVal))
print("Autoencoder Training completed")

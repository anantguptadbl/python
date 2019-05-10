#import pandas as pd
#data=pd.read_csv('hotel-review/train.csv')

import nltk

validPosTags=['JJ','JJR','JJS','NN','RB','RBR','RBS','VB','VBD','VBG','VBZ']

def removeUnncessaryTags(x):
    # For ech sentnece in a row
    # For each word in a sentence
    validSentences=[]
    for curSentence in x.split('.'):
        validWords=[]
        for curWord in curSentence.split(' '):
            if(len(curWord.strip()) > 2):
                if(nltk.pos_tag([curWord.lower()])[0][1] in validPosTags):
                    validWords.append(curWord.lower())
        validSentences.append(validWords)
    return(validSentences)
            
                
data=data[0:10000]
data['data']=data['Description'].map(lambda x : removeUnncessaryTags(x))
data.head(5)


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
        self.hidden1_1 = torch.autograd.variable(torch.randn(1,self.batch_size1,self.hiddenDim))
        self.hidden1_2 = torch.autograd.variable(torch.randn(1,self.batch_size1,self.hiddenDim))
        
         # ATTENTION WEIGHTS WORD LEVEL
        self.attentionWeightsWords=torch.autograd.variable(torch.randn(self.hiddenDim,self.step_size1))
        #self.attentionWeightsWords=torch.autograd.variable(torch.randn(self.step_size1,self.hiddenDim))
        #self.attentionWeightsSentence=torch.autograd.variable(torch.randn(self.inputDim,self.step_size))
        
    def forward(self,inputs):
        print("the size of the inputs is {}".format(inputs.size()))
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
            #print(self.attentionWeightsWordsNormalized)
        
            # ITERATING THROUGH EACH WORD SEQUENCE
            print(len(self.wordEncodingHiddenList))
            print(self.wordEncodingHiddenList[0].shape)
            print(torch.stack(self.wordEncodingHiddenList).view(self.step_size1,self.hiddenDim).size())
            print(self.attentionWeightsWordsNormalized.size())
            self.wordEncodingHiddenList1=torch.stack(self.wordEncodingHiddenList).view(self.step_size1,self.hiddenDim)
        
            self.stepAttentionOutput=self.wordEncodingHiddenList1 * torch.t(self.attentionWeightsWordsNormalized)
            print(self.stepAttentionOutput.size())
          
            # NOW WE WILL USE THIS TO ARRIVE AT A SINGLE VECTOR FOR EACH SENTENCE
            print("After the calculations is done, we have got the vector for a single sentence")
            print(self.stepAttentionOutput.size())
            print("We will now start the attention for the sentence level stuff")
        
            self.out1,(self.hidden1_1,self.hidden1_2) = self.lstm1(x.view(1,self.batch_size1,self.inputDim),(self.hidden1_1,self.hidden1_2))
            self.wordEncodingHiddenList.append(self.hidden1_1)
        
        #return self.attentionOutput
        return torch.stack(self.stepAttentionOutput)
    
def lossCalc(x,y):
    return torch.sum(torch.add(x,-y)).pow(2) 

# Model Object
inputDim=20
hiddenDim=40
outputDim=40
epochRange=100

    
# LSTM Configuration
numRows=2500  # This denotes the number of rows. Each row consits of the number of inputElements
numSentences=5       # This is the number of rows that will be used for gradient update
numWords=10         # This is the number of LSTM cells
batch_size1=1
batch_size2=5
totalBatches=numRows/batch_size1

model=LSTMHierarchialAttention(inputDim,hiddenDim,outputDim,batch_size1,batch_size2,numWords,numSentences)
#loss = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(),lr=0.0001,amsgrad=True,weight_decay=0.85)

# Input Data
X=np.random.rand(numRows * numSentences * numWords,inputDim)

# Resizing
X=X.reshape(totalBatches,numSentences,numWords,inputDim)

for epoch in range(epochRange):
    lossVal=0
    for curBatch in range(totalBatches):
        model.zero_grad()
        dataInput=torch.autograd.Variable(torch.Tensor(X[curBatch]))
        dataOutput=model(dataInput)
        print(dataOutput.size())
        loss=lossCalc(dataOutput,dataInput.view(step_size*batch_size1,-1))
        loss.backward()
        lossVal = lossVal + loss
        optimizer.step()
    if(epoch % 100==0):
        print("For epoch {}, the loss is {}".format(epoch,lossVal))
print("Autoencoder Training completed")

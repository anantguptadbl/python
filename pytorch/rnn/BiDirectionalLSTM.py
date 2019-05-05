import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy

class BiDirectionalLSTM(nn.Module):
    def __init__(self,inputDim,hiddenDim,outputDim,batch_size,step_size):
        super(BiDirectionalLSTM,self).__init__()
        self.inputDim=inputDim
        self.hiddenDim=hiddenDim
        self.outputDim=outputDim
        self.batch_size=batch_size
        self.step_size=step_size
        torch.manual_seed(1)
        # input_size – The number of expected features in the input x
        # hidden_size – The number of features in the hidden state h
        # num_layers – Number of recurrent layers
        # FORWARD LSTM
        self.lstm1=nn.LSTM(self.inputDim,self.hiddenDim)
        # BACKWARD LSTM
        self.lstm2=nn.LSTM(self.inputDim,self.hiddenDim)
        # Hidden state is a tuple of two states, so we will have to initialize two tuples
        # h_0 of shape (num_layers * num_directions, batch, hidden_size)
        # c_0 of shape (num_layers * num_directions, batch, hidden_size)
        self.hidden1_1 = torch.autograd.variable(torch.randn(1,self.batch_size,self.hiddenDim))
        self.hidden1_2 = torch.autograd.variable(torch.rand(1,self.batch_size,self.hiddenDim))
        self.hidden2_1 = torch.autograd.variable(torch.randn(1,self.batch_size,self.hiddenDim))
        self.hidden2_2 = torch.autograd.variable(torch.randn(1,self.batch_size,self.hiddenDim))
        self.linearModel=nn.Linear(self.hiddenDim,self.outputDim)

    def forward(self,inputs1,inputs2):
        self.fullDataOutput=[]
        self.out1List=[]
        self.out2List=[]
        # Forward LSTM
        for x in inputs1:
            out1,(self.hidden1_1,self.hidden1_2) = self.lstm1(x.view(1,self.batch_size,self.inputDim),(self.hidden1_1,self.hidden1_2))
            self.out1List.append(out1)
        # Backward LSTM
        for x in inputs2:
            out2,(self.hidden2_1,self.hidden2_2) = self.lstm2(x.view(1,self.batch_size,self.inputDim),(self.hidden2_1,self.hidden2_2))
            self.out2List.append(out2)
    
        self.out1=torch.stack(self.out1List).view(-1,self.batch_size,self.hiddenDim)
        self.out2=torch.stack(self.out2List).view(-1,self.batch_size,self.hiddenDim)
    
        self.hidden1_1=self.hidden1_1.detach()
        self.hidden1_2=self.hidden1_2.detach()
        self.hidden2_1=self.hidden2_1.detach()
        self.hidden2_2=self.hidden2_2.detach()
        
        # Combining the two outputs       
        self.out=torch.add(self.out1,self.out2)
        
        # Linear Output
        self.outLinear=self.linearModel(self.out)
        self.fullDataOutput.append(self.outLinear)
        return torch.stack(self.fullDataOutput).view(-1,self.outputDim)
    
    def predict(self,input1,input2):
        numRows=input1.size()[1]
        hidden1_1 = torch.autograd.variable(torch.randn(1,numRows,self.inputDim))
        hidden1_2 = torch.autograd.variable(torch.randn(1,numRows,self.inputDim))
        hidden2_1 = torch.autograd.variable(torch.randn(1,numRows,self.inputDim))
        hidden2_2 = torch.autograd.variable(torch.randn(1,numRows,self.inputDim))
        out1,hidden1 = self.lstm1(input1,(hidden1_1,hidden1_2))
        out2,hidden2 = self.lstm2(input2,(hidden2_1,hidden2_2))
        out=torch.add(out1,out2)
        result=self.linearModel(out)
        return result
        
def lossCalc(x,y):
    return torch.sum(torch.add(x,-y)).pow(2) 

# Model Object
inputDim=3
hiddenDim=3
outputDim=2

# LSTM Configuration
batch_size=10
step_size=5
totalBatches=100/(step_size*batch_size)
epochRange=1000
numRows=100

model=BiDirectionalLSTM(inputDim,hiddenDim,outputDim,batch_size,step_size)
#loss = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Input Data
X=np.random.rand(numRows,inputDim)
Y=np.random.rand(numRows,outputDim)

# Resizing
X=X.reshape(totalBatches,step_size,batch_size,inputDim)
Y=Y.reshape(totalBatches,step_size,batch_size,outputDim)

for epoch in range(epochRange):
    for curBatch in range(numRows/(step_size*batch_size)):
        dataInput1=torch.Tensor(X[curBatch])
        dataInput2=torch.Tensor(copy.deepcopy(X[curBatch][::-1]))
        #dataInput=[torch.Tensor(x) for x in dataInput]
        
        dataY=torch.Tensor(Y[curBatch])
        #dataY=[torch.Tensor(x) for x in dataInput]
        dataOutput=model(dataInput1,dataInput2)
        loss=lossCalc(dataOutput,dataY.view(-1,outputDim))
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
    if(epoch % 100==0):
        print("For epoch {}, the loss is {}".format(epoch,loss))
        
# PREDICTION
# The shape of input data 1 * step_size * inputDim
data=np.array([[0,0,1],[0,1,0],[0,0,0],[1,0,0],[1,1,0]]).reshape(5,1,3)
prediction = model.predict(torch.Tensor(data),torch.Tensor(copy.deepcopy(data[::-1]) ))
print(prediction)

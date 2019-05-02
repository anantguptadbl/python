import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class LSTMSimple(nn.Module):
    def __init__(self,inputDim,hiddenDim,outputDim,batch_size,step_size):
        super(LSTMSimple,self).__init__()
        self.inputDim=inputDim
        self.hiddenDim=hiddenDim
        self.outputDim=outputDim
        self.batch_size=batch_size
        self.step_size=step_size
        torch.manual_seed(1)
        # input_size – The number of expected features in the input x
        # hidden_size – The number of features in the hidden state h
        # num_layers – Number of recurrent layers
        self.lstm=nn.LSTM(self.inputDim,self.hiddenDim)
        # Hidden state is a tuple of two states, so we will have to initialize two tuples
        # h_0 of shape (num_layers * num_directions, batch, hidden_size)
        # c_0 of shape (num_layers * num_directions, batch, hidden_size)
        self.hidden1 = torch.autograd.variable(torch.randn(1,self.batch_size,self.hiddenDim))
        self.hidden2 = torch.autograd.variable(torch.rand(1,self.batch_size,self.hiddenDim))
        self.linearModel=nn.Linear(self.hiddenDim,self.outputDim)

    def forward(self,inputs):
        self.fullDataOutput=[]
        # input of shape (seq_len, batch, input_size)
        # h_0 of shape (num_layers * num_directions, batch, hidden_size)
        # c_0 of shape (num_layers * num_directions, batch, hidden_size)
        self.out,self.hidden = self.lstm(inputs,(self.hidden1,self.hidden2))
        self.hidden1=self.hidden1.detach()
        self.hidden2=self.hidden2.detach()
        self.outLinear=self.linearModel(self.out)
        self.fullDataOutput.append(self.outLinear)
        return torch.stack(self.fullDataOutput).view(-1,self.outputDim)
    
    def predict(self,inputs):
        numRows=inputs.size()[1]
        hidden1 = torch.autograd.variable(torch.randn(1,numRows,self.inputDim))
        hidden2 = torch.autograd.variable(torch.rand(1,numRows,self.inputDim))
        out,hidden = self.lstm(inputs,(hidden1,hidden2))
        return out
        
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


model=LSTMSimple(inputDim,hiddenDim,outputDim,batch_size,step_size)
#loss = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Input Data
X=np.random.rand(100,inputDim)
Y=np.random.rand(100,outputDim)

# Resizing
X=X.reshape(totalBatches,step_size,batch_size,inputDim)
Y=Y.reshape(totalBatches,step_size,batch_size,outputDim)

for epoch in range(100):
    for curBatch in range(100/(step_size*batch_size)):
        dataInput=torch.Tensor(X[curBatch])
        #dataInput=[torch.Tensor(x) for x in dataInput]
        
        dataY=torch.Tensor(Y[curBatch])
        #dataY=[torch.Tensor(x) for x in dataInput]
        
        dataOutput=model(dataInput)
        loss=lossCalc(dataOutput,dataY.view(-1,outputDim))
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
    if(epoch % 100==0):
        print("For epoch {}, the loss is {}".format(epoch,loss))
        
# PREDICTION
testInput=torch.Tensor(np.array([[0,0,1],[0,1,0],[0,0,0],[1,0,0],[1,1,0]]).reshape(5,1,3))
prediction = model.predict(testInput)
print(prediction)

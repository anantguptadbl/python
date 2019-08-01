import os
import numpy as np

import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

learningRate=0.01
num_epochs=100

def convertToSparseFloatTensor(inpArr):
    inpArr=scipy.sparse.coo_matrix(inpArr)
    i = torch.LongTensor(np.vstack((inpArr.row, inpArr.col)).astype(np.int32))
    v = torch.FloatTensor(inpArr.data.astype(np.float32))
    return(torch.sparse.FloatTensor(i, v, torch.Size(inpArr.shape)).to_dense())

class linearRegressionOLS(nn.Module):
    def __init__(self):
        super(linearRegressionOLS,self).__init__()
        self.linearModel=nn.Linear(10,1)
        
    def forward(self,x):
        x = self.linearModel(x)
        return x
    
model=linearRegressionOLS()
criterion=nn.MSELoss()
optimizer=torch.optim.SGD(model.parameters(),lr=learningRate)

X=np.random.rand(100,10).astype(np.float32)
Y=np.random.randint(2,size=(100)).reshape(100,1).astype(np.float32)

for epoch in range(num_epochs):
    inputVal=Variable(convertToSparseFloatTensor(X))
    outputVal=Variable(convertToSparseFloatTensor(Y))
    # In a gradient descent step, the following will now be performing the gradient descent now
    optimizer.zero_grad()
    # We will now setup the model
    dataOutput = model(inputVal)
    # We will now define the loss metric
    loss = criterion(dataOutput, outputVal)
    # We will perform the backward propagation
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, num_epochs, loss))        

# Final weights of the linear Regression Model
coeff=model.linearModel.weight.data.numpy()
print("the final coeff are {}".format(coeff))

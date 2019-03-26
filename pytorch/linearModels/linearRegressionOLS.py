# Linear Regression OLS
import os
import numpy as np

import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

class linearRegressionOLS(nn.Module):
    def __init__(self):
        super(linearRegressionOLS,self).__init__()
        self.linearModel=nn.Linear(10,1)
        
    def forward(self,x):
        x = self.linearModel(x)
        return x
    
model=linearRegressionOLS()
criterion=nn.MSELoss()
optimizer=torch.optim.SGD(model.parameters(),lr=learning_rate)

X=np.random.rand(100,10).astype(np.float32)
Y=np.random.randint(2,size=(100)).reshape(100,1).astype(np.float32)

for epoch in range(num_epochs):
    inputVal=Variable(torch.from_numpy(X))
    outputVal=Variable(torch.from_numpy(Y))
    # In a gradient descent step, the following will now be performing the gradient descent now
    optimizer.zero_grad()
    # We will now setup the model
    dataOutput = model(dataInput)
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

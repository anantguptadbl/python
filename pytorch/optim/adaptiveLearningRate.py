import os
import numpy as np

import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

num_epochs = 100
learning_rate = 1e-3

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(10,5),
            nn.ReLU(True),
            nn.Linear(5,3),
            nn.ReLU(True))
        self.decoder = nn.Sequential(
            nn.Linear(3, 5),
            nn.ReLU(True),
            nn.Linear(5,10),
            nn.ReLU(True)
            )
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


model = autoencoder()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True)

data=np.random.rand(100,10).astype(np.float32)
dataInput=Variable(torch.from_numpy(data))
running_loss=0
for epoch in range(num_epochs):
    dataOutput = model(dataInput)
    loss = criterion(dataOutput, dataInput)
    running_loss += loss.item()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, num_epochs, loss))
    lr_scheduler.step(running_loss)
        
# How to get the weights of the Layers
#FirstLayerWeights=np.array(model.encoder[0].weight.tolist()).T

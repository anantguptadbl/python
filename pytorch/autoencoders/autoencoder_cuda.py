import os
import numpy as np
import time
import torch
import torchvision
from torch import nn
from torch.autograd import Variable

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(1000,200).cuda(),
            nn.ReLU(True).cuda(),
            nn.Linear(200,30).cuda(),
            nn.ReLU(True).cuda())
        self.decoder = nn.Sequential(
            nn.Linear(30, 200).cuda(),
            nn.ReLU(True).cuda(),
            nn.Linear(200,1000).cuda(),
            nn.ReLU(True).cuda()
            )
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

model = autoencoder().cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

startTime=time.time()
data=np.random.rand(10000,1000).astype(np.float32)
dataInput=Variable(torch.from_numpy(data)).cuda()
for epoch in range(num_epochs):
    dataOutput = model(dataInput)
    loss = criterion(dataOutput, dataInput)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, num_epochs, loss))
timeTaken=time.time()-startTime
print("Time Taken is {0}".format(timeTaken))

import os
import numpy as np
import time
import torch
import torchvision
from torch import nn
from torch.autograd import Variable

import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster, datasets, mixture

num_features=100
num_rows=10000
blobs = datasets.make_blobs(n_samples=num_rows, n_features=num_features, random_state=8, centers=8)
X = blobs[0].astype(np.float32)
labels = blobs[1]

torch.manual_seed(42)

class autoencoder(nn.Module):
    def __init__(self, num_features):
        super(autoencoder, self).__init__()
        self.encoder1 = nn.Sequential(
            nn.Linear(num_features,200),
            nn.BatchNorm1d(200),
            nn.ReLU(True).cuda(),
            nn.Linear(200,num_features),
            nn.BatchNorm1d(num_features),
            nn.ReLU(True)
        )
        
        self.encoder2 = nn.Sequential(
            nn.Linear(num_features, 10),
            nn.BatchNorm1d(10),
            nn.ReLU(True)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(10, 200),
            nn.BatchNorm1d(200),
            nn.ReLU(True).cuda(),
            nn.Linear(200,num_features),
            nn.BatchNorm1d(num_features),
            nn.ReLU(True)
            )
    def forward(self, x):
        x = self.encoder1(x)
        x = self.encoder2(x)
        x = self.decoder(x)
        return x


num_epochs=10000
learning_rate=0.001
model = autoencoder(num_features).cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

startTime=time.time()
dataInput=Variable(torch.from_numpy(X)).cuda()

for epoch in range(num_epochs):
    dataOutput = model(dataInput)
    loss = criterion(dataOutput, dataInput)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 500 == 0:
        print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, num_epochs, loss))
timeTaken=time.time()-startTime
print("Time Taken is {0}".format(timeTaken))

import os
import numpy as np
import time
import torch
import torchvision
from torch import nn
from torch.autograd import Variable

import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster, datasets, mixture

num_features=100
num_rows=10000
blobs = datasets.make_blobs(n_samples=num_rows, n_features=num_features, random_state=8, centers=8)
X = blobs[0].astype(np.float32)
labels = blobs[1]

torch.manual_seed(42)

class autoencoder(nn.Module):
    def __init__(self, num_features):
        super(autoencoder, self).__init__()
        self.encoder1 = nn.Sequential(
            nn.Linear(num_features,200),
            nn.BatchNorm1d(200),
            nn.ReLU(True).cuda(),
            nn.Linear(200,num_features),
            nn.BatchNorm1d(num_features),
            nn.ReLU(True)
        )
        
        self.encoder2 = nn.Sequential(
            nn.Linear(num_features, 10),
            nn.BatchNorm1d(10),
            nn.ReLU(True)
        )
        self.decoder1 = nn.Sequential(
            nn.Linear(10, 200),
            nn.BatchNorm1d(200),
            nn.ReLU(True),
            nn.Linear(200, 10),
            nn.BatchNorm1d(10),
            nn.ReLU(True)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(10,num_features),
            nn.BatchNorm1d(num_features),
            nn.ReLU(True)
            )
    def forward(self, x):
        x = self.encoder2(self.encoder1(x) + x)
        x = self.decoder(self.decoder1(x) + x)
        return x


num_epochs=10000
learning_rate=0.001
model = autoencoder(num_features).cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

startTime=time.time()
dataInput=Variable(torch.from_numpy(X)).cuda()
for epoch in range(num_epochs):
    dataOutput = model(dataInput)
    loss = criterion(dataOutput, dataInput)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 500 == 0:
        print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, num_epochs, loss))
timeTaken=time.time()-startTime
print("Time Taken is {0}".format(timeTaken))

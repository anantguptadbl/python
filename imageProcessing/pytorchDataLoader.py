import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as data
import torchvision
from torchvision import transforms
import torchvision.models as models

# VGG Model
vgg16 = models.vgg16(pretrained=True)

# Requires Grad False
# Freezing the first few layers of the alexnet model
for moduleIndex in range(30):
    for param in vgg16.features._modules[str(moduleIndex)].parameters():
        param.requires_grad=False
        print(param.requires_grad)
        
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
   ])

transform_test = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
 
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform_train, download=True)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test) 
print("Cell Execution Completed")   

import torch.optim as optim

class Transfer(nn.Module):
    def __init__(self):
        super(Transfer, self).__init__()
        self.vgg16=vgg16.features
        self.layer1=nn.Linear(512*7*7,10)
        self.smax=nn.Softmax(dim=1)
        
    def forward(self,x):
        x=self.vgg16(x)
        x=self.layer1(x.view(-1,512*7*7))
        x=self.smax(x)
        return(x)
    
model=Transfer()
optimizer = optim.SGD(model.parameters(), lr=0.001,momentum=0.6)
criterion = nn.CrossEntropyLoss()

# Config
epochs=10
batchSize=64

# Data Loader
import torch.utils.data as data
train_data_loader = data.DataLoader(trainset, batch_size=batchSize, shuffle=True,  num_workers=4)
test_data_loader  = data.DataLoader(testset, batch_size=batchSize, shuffle=True, num_workers=4) 

for epoch in range(10):     
    totalLoss=0
    for step, (x, y) in enumerate(train_data_loader):
        model.zero_grad()
        output = model(x)      
        loss = criterion(output, y)   
        totalLoss=totalLoss + loss.item()         
        loss.backward()                 
        optimizer.step()
    if(epoch%1==0):
        print("Epoch {0} TotalLoss {1}".format(epoch,totalLoss))

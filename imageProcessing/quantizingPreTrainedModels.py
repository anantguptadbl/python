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
import pickle

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

class LimitDataset(data.Dataset):
    def __init__(self, dataset, n):
        self.dataset = dataset
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return self.dataset[i]


trainset=LimitDataset(torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform_train, download=True),100)
testset=LimitDataset(torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform_test, download=True),100)
#trainset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform_train, download=False)
#testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test) 
print("Cell Execution Completed")   

import torch.optim as optim

torch.manual_seed(42)

class Flatten(nn.Module):
    def forward(self, x):
        return x.reshape(-1, 512 * 7 * 7)

class Transfer(nn.Module):
    def __init__(self):
        super(Transfer, self).__init__()
        self.model1=nn.Sequential \
        ( \
        vgg16.features, \
        Flatten(), \
        nn.Linear(512*7*7,10) \
        ) 
        self.smax=nn.Softmax(dim=1)
        
    def forward(self,x):
        x=self.model1(x)
        x=self.smax(x)
        return(x)
    
model=Transfer()
optimizer = optim.SGD(model.parameters(), lr=0.01,momentum=0.6)
criterion = nn.CrossEntropyLoss()

# Config
epochs=2
batchSize=8

# Data Loader
import torch.utils.data as data
train_data_loader = data.DataLoader(trainset, batch_size=batchSize, shuffle=False,  num_workers=0)
test_data_loader  = data.DataLoader(testset, batch_size=batchSize, shuffle=False, num_workers=0) 

# Model Training
model=Transfer()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
for epoch in range(epochs):     
    totalLoss=0
    for step, (x, y) in enumerate(train_data_loader):
        if(step <= 100):
            model.zero_grad()
            output = model(x)      
            loss = criterion(output, y)   
            totalLoss=totalLoss + loss.item()         
            loss.backward()                 
            optimizer.step()
    if(epoch%1==0):
        print("Epoch {0} Step {1} TotalLoss {2}".format(epoch,step,totalLoss))
        
model=model.to('cpu')
torch.save(model,"torchNonQuantizedModel")
print(os.path.getsize("torchNonQuantizedModel")/1e6)  # 79.003

print("Cell Execution Completed")

# POST TRAINING STATIC QUANTIZATION
# Load the non quantized model
model=torch.load("torchNonQuantizedModel")

# Fuse Model
model.eval()

# Eval Model
#model.fuse_model()

# Setting up the configuration
model.qconfig = torch.quantization.default_qconfig
print(model.qconfig)
# Model Preparation : This will inject the observers
torch.quantization.prepare(model, inplace=True)

for curModule in model._modules['model1']._modules['0']:
    if(type(curModule)==torch.nn.modules.activation.ReLU):
        curModule.inplace=False
        
# Checking
for curModule in model._modules['model1']._modules['0']:
    if(type(curModule)==torch.nn.modules.activation.ReLU):
        print(curModule.inplace)
        
# Calibration of the model
print("Calibrating the model")
num_calibration_batches=10
# Evaluation where we are passing data to the model after the observers have been injected through prepare function
with torch.no_grad():
    for step, (x, y) in enumerate(train_data_loader):
        yHat = model(x)
        loss = criterion(yHat, y)

# Quantization of the model
newModel=torch.quantization.convert(model,inplace=False)
torch.save(newModel.state_dict(), "torchQuantizedModel")
print(os.path.getsize("torchQuantizedModel")/1e6) # 19.771906

# Testing with Quantized Model
dqm = torch.nn.quantized.DeQuantize()
totalLoss=0
for step, (x, y) in enumerate(train_data_loader):
    if(step <= 100):
        newModel.zero_grad()
        scale, zero_point = 1e-4, 128
        q = torch.quantize_per_tensor(x, scale, zero_point, torch.quint8)
        output = newModel.model1(q)      
        dequantized = dqm(output)
        output=model.smax(dequantized)
        loss = criterion(output, y)   
        totalLoss=totalLoss + loss.item()         
        print(totalLoss)
        
print("Cell Execution Completed")

# IMPORTS
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

print("Import Cell Execution Completed")

# MODEL VGG16 + ADDITIONAL LAYER

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

# QUANTIZATION AWARE TRAINING

# https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html#quantization-aware-trainin
# Load the non quantized model
model=torch.load("torchNonQuantizedModel")

# Fuse Modules in the model
model=torch.load("torchNonQuantizedModel")
torch.quantization.fuse_modules(model.model1[0], ['0','1'], inplace=True)
torch.quantization.fuse_modules(model.model1[0], ['2','3'], inplace=True)
torch.quantization.fuse_modules(model.model1[0], ['5','6'], inplace=True)
torch.quantization.fuse_modules(model.model1[0], ['7','8'], inplace=True)
torch.quantization.fuse_modules(model.model1[0], ['10','11'], inplace=True)
torch.quantization.fuse_modules(model.model1[0], ['12','13'], inplace=True)
torch.quantization.fuse_modules(model.model1[0], ['14','15'], inplace=True)
torch.quantization.fuse_modules(model.model1[0], ['17','18'], inplace=True)
torch.quantization.fuse_modules(model.model1[0], ['19','20'], inplace=True)
torch.quantization.fuse_modules(model.model1[0], ['21','22'], inplace=True)
torch.quantization.fuse_modules(model.model1[0], ['24','25'], inplace=True)
torch.quantization.fuse_modules(model.model1[0], ['26','27'], inplace=True)
torch.quantization.fuse_modules(model.model1[0], ['28','29'], inplace=True)

# Set Config
optimizer = torch.optim.SGD(model.parameters(), lr = 0.0001)
model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')

# Preparation
torch.quantization.prepare_qat(model, inplace=True)

# Other modeling stuff
optimizer = optim.SGD(model.parameters(), lr=0.01,momentum=0.6)
criterion = nn.CrossEntropyLoss()
numBatches=16
numBatchesTest=8

print("Cell Execution Completed")

for curEpoch in range(3):
	# Training Data and generate loss
	totalLoss=0
	for step, (x, y) in enumerate(train_data_loader):
		model.zero_grad()
		output = model(x)      
		loss = criterion(output, y)   
		totalLoss=totalLoss + loss.item()         
		loss.backward()                 
		optimizer.step()
	print("Epoch {0} Step {1} TotalLoss {2}".format(epoch,step,totalLoss))
	if(curEpoch > 2):
		# Freeze quantizer parameters
		model.apply(torch.quantization.disable_observer)
	if(curEpoch > 2):
		# Freeze batch norm mean and variance estimates
		model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
	# Check the accuracy after each epoch
	quantized_model = torch.quantization.convert(model.eval(), inplace=False)
	quantized_model.eval()
 
m = torch.jit.script(quantized_model)
torch.jit.save(m, "quantizationAwareTrainedModel.pt")
quantizedModelLoaded=torch.jit.load("quantizationAwareTrainedModel.pt")

with torch.no_grad():  
  totalLoss=0
  for step, (x, y) in enumerate(test_data_loader):
    model.zero_grad()
    xQuant = torch.quantize_per_tensor(x, scale=1e-3, zero_point=128,dtype=torch.quint8)
    output = quantized_model.model1(xQuant)
    output = torch.dequantize(output)
    output = quantized_model.smax(output)
    loss = criterion(output, y)   
    totalLoss=totalLoss + loss.item()     
print("The total loss is {0}".format(totalLoss))    

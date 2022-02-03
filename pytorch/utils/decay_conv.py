# SPATIAL NEURONS
# SIMULATING CONVOLUTION NEURAL NETWORK
# DATASET : IMAGE CLASSIFIER

import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd.functional as F1
import torch.optim as optim
from torch.autograd import Variable
import torchvision
import matplotlib.pyplot as plt

batch_size_train = 64
batch_size_test = 1000
log_interval = 10

random_seed = 42
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

mnist_data = torchvision.datasets.MNIST('./mnist_files/', train=True, download=False,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ]))

fig = plt.figure()
for i in range(6):
    plt.subplot(2,3,i+1)
    plt.tight_layout()
    plt.imshow(mnist_data.train_data[i], cmap='gray', interpolation='none')
    plt.title("Ground Truth: {}".format(mnist_data.train_labels[i]))
    plt.xticks([])
    plt.yticks([])
fig

mnist_data.train_data.data = mnist_data.train_data/255.
mnist_data.train_data.data = mnist_data.train_data.data.reshape(-1, 1, 28, 28)

class Normal_Net(nn.Module):
    def __init__(self):
        super(Normal_Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x)
    
model = Normal_Net()

learning_rate = 0.01
momentum = 0.5

optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
num_batches = int(mnist_data.train_data.shape[0]/batch_size_train)
num_batches = 20
n_epochs = 100

print("Cell Executed")

for cur_epoch in range(n_epochs):
    total_loss=0
    for cur_batch in range(num_batches):
        optimizer.zero_grad()
        output = model(mnist_data.train_data[cur_batch*batch_size_train: (cur_batch+1)*batch_size_train])
        loss = F.nll_loss(output, mnist_data.train_labels[cur_batch*batch_size_train: (cur_batch+1)*batch_size_train])
        total_loss = total_loss + loss.item()
        loss.backward()
        optimizer.step()
    if cur_epoch % 10==0:
        print("Epoch {0} Loss is {1} Total Loss {2}".format(cur_epoch, loss.item(), total_loss))
        
 # CUSTOM NEURON

class SpatialNeuron(nn.Module):
    def __init__(self):
        super(SpatialNeuron, self).__init__()
        self.weights = nn.Parameter(torch.rand(4), requires_grad=True)
        self.bias = nn.Parameter(torch.rand(1), requires_grad=True)
        
    def forward(self, x):
        x = torch.matmul(x.reshape(-1,4), self.weights) + self.bias
        return x
        
class CustomConvolve(nn.Module):
    def __init__(self, image_width, image_height):
        super(CustomConvolve, self).__init__()
        self.num_neurons = (image_width-1) * (image_height-1)
        self.image_width = image_width
        self.image_height = image_height
        self.spatial_neurons = nn.ModuleList([SpatialNeuron() for x in range(self.num_neurons)])
        
    def forward(self, x):
        output = torch.zeros(x.size()[0], x.size()[1], self.image_width-1, self.image_height-1)
        for cur_channel in range(x.size()[1]):
            for cur_width_indexer in range(1, self.image_width-1):
                for cur_height_indexer in range(1, self.image_height-1):
                    #print(len(self.spatial_neurons[
                    #    (self.image_width-1)*cur_width_indexer + cur_height_indexer
                    #](x[:, cur_channel, cur_width_indexer-1:cur_width_indexer+1, cur_height_indexer-1:cur_height_indexer+1])))
                    #print(output[:, cur_channel, cur_width_indexer, cur_height_indexer].size())
                    output[:, cur_channel, cur_width_indexer, cur_height_indexer] = self.spatial_neurons[
                        (self.image_width-1)*cur_width_indexer + cur_height_indexer
                    ](x[:, cur_channel, cur_width_indexer-1:cur_width_indexer+1, cur_height_indexer-1:cur_height_indexer+1])
        return output
    
class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.layer1 = CustomConvolve(image_width=28, image_height=28)
        self.layer2 = CustomConvolve(image_width=27, image_height=27)
        self.layer3 = CustomConvolve(image_width=26, image_height=26)
        self.layer4 = CustomConvolve(image_width=25, image_height=25)
        self.layer5 = CustomConvolve(image_width=24, image_height=24)
        #self.fc1 = nn.Linear(23*23, 50)
        #self.fc2 = nn.Linear(50, 10)
        self.fc1 = DecayLinear(length=50, neuron_length=23*23)
        self.fc2 = DecayLinear(length=10, neuron_length=50)
        
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        x = F.relu(self.layer5(x))
        x = x.view(-1, 23*23)
        x = self.fc1(x)
        x = self.fc2(x)
        x = F.log_softmax(x)
        return x
    
model = CustomModel()

learning_rate = 0.001
momentum = 0.5

optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
num_batches = int(mnist_data.train_data.shape[0]/batch_size_train)
num_batches = 20
n_epochs = 100

for cur_epoch in range(n_epochs):
    total_loss=0
    for cur_batch in range(num_batches):
        optimizer.zero_grad()
        output = model(mnist_data.train_data[cur_batch*batch_size_train: (cur_batch+1)*batch_size_train])
        loss = F.nll_loss(output, mnist_data.train_labels[cur_batch*batch_size_train: (cur_batch+1)*batch_size_train])
        total_loss = total_loss + loss.item()
        loss.backward()
        optimizer.step()
    if cur_epoch % 1==0:
        print("Epoch {0} Loss is {1} Total Loss {2}".format(cur_epoch, loss.item(), total_loss))

import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd.functional as F1
import torch.optim as optim
from torch.autograd import Variable

# DATA
torch.manual_seed(42)
input_features = 10

X = torch.rand(1000, input_features)
y = torch.rand(1000)

class NormalModel(nn.Module):
    def __init__(self):
        super(NormalModel, self).__init__()
        self.layer1 = nn.Linear(input_features, 10)
        self.layer2 = nn.Linear(10, 1)
        
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.layer2(x)
        return x
    
model = NormalModel()
    
loss = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
    
for cur_epoch in range(100):
    optimizer.zero_grad()
    output = model(X)
    final_loss = loss(output, y)
    final_loss.backward()
    optimizer.step()
    print("Epoch {0} Loss is {1}".format(cur_epoch, final_loss.item()))
    
# CUSTOM NEURON

class DecayNeuron(nn.Module):
    def __init__(self, length):
        super(DecayNeuron, self).__init__()
        self.weights = nn.Parameter(torch.rand(length), requires_grad=True)
        self.bias = nn.Parameter(torch.rand(1), requires_grad=True)
        # 2 rest iterations
        self.current_active = torch.zeros(length, 3)
        self.init_positions = np.array([[i,x] for i,x in enumerate(np.random.randint(3, size=(length, )))])
        self.current_active[self.init_positions[:, 0], self.init_positions[:, 1]] = 1
        
    def forward(self, x):
        x = torch.matmul(x, current_active[:, 2] * self.weights) + self.bias
        # We will roll to make it inactive for the next two iterations
        self.current_active = torch.roll(self.current_active, 1, 1)
        return x
        
class DecayLinear(nn.Module):
    def __init__(self, length, neuron_length):
        super(DecayLinear, self).__init__()
        self.neurons = nn.ModuleList([DecayNeuron(neuron_length) for x in range(length)])
        
    def forward(self, x):
        x = [cur_neuron(x) for cur_neuron in self.neurons]
        x = torch.stack(x)
        x = torch.transpose(x, 0, 1)
        x = F.relu(x)
        return x
    
class DecayModel(nn.Module):
    def __init__(self):
        super(DecayModel, self).__init__()
        self.layer1 = DecayLinear(length=10, neuron_length=input_features)
        self.layer2 = DecayLinear(length=1, neuron_length=10)
        
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.layer2(x)
        return x

model = DecayModel()
    
loss = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
    
for cur_epoch in range(100):
    optimizer.zero_grad()
    output = model(X)
    final_loss = loss(output, y)
    final_loss.backward()
    optimizer.step()
    print("Epoch {0} Loss is {1}".format(cur_epoch, final_loss.item()))

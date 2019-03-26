# RNN
import torch
import torchvision
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F

class RNNSimple(nn.Module):
    def __init__(self,inputFeatures,numNeurons):
        super(RNNSimple,self).__init__()
        self.weight_input=torch.randn(inputFeatures,numNeurons)
        self.weight_hidden=torch.randn(numNeurons,numNeurons)
        self.bias=torch.zeros(1,numNeurons)
        
    def forward(self,X0,X1):
        self.Y0=torch.tanh(torch.mm(X0,self.weight_input) + self.bias)
        self.Y1=torch.tanh(torch.mm(self.Y0,self.weight_hidden) + torch.mm(X1,self.weight_input) + self.bias )
        return self.Y0,self.Y1
    
N_INPUT = 4
N_NEURONS = 1
X0_batch = torch.tensor([[0,1,2,0], [3,4,5,0], [6,7,8,0], [9,0,1,0]],
                        dtype = torch.float) #t=0 => 4 X 4

X1_batch = torch.tensor([[9,8,7,0], [0,0,0,0], [6,5,4,0], [3,2,1,0]],
                        dtype = torch.float) #t=1 => 4 X 4

print(X0_batch.shape)
print(X1_batch.shape)

model = RNNSimple(N_INPUT, N_NEURONS)
Y0_val, Y1_val = model(X0_batch, X1_batch)
print(Y0_val.shape)
print(Y1_val.shape)

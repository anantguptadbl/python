import pandas as pd
import numpy as np
import matplotlib.image as mpimg 
import matplotlib.pyplot as plt 
import numpy as np
import copy
import pandas as pd
import cv2
import pandas as pd
# Using a pretrained model
from torchvision import models
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from collections import OrderedDict 
import numpy as np
import gc
import cv2


# ALEXNET MODEL : VERY BAD RESULTS
alexnet = models.alexnet(pretrained=True)

# Freezing the first few layers of the alexnet model
for moduleIndex in ['0','3','6']:
    for param in alexnet.features._modules[moduleIndex].parameters():
        param.requires_grad=False
        print(param.requires_grad)

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(-1, 256 *2 *2)
    
class Transfer(nn.Module):
    def __init__(self):
        super(Transfer, self).__init__()
        self.SequentialModel1 = nn.Sequential(OrderedDict([
            ('conv1',alexnet.features._modules['0']),
            ('relu1',alexnet.features._modules['1']),
            ('pool1',alexnet.features._modules['2']),
            ('conv2',alexnet.features._modules['3']),
            ('relu2',alexnet.features._modules['4']),
            ('pool2',alexnet.features._modules['5']),
            ('conv3',alexnet.features._modules['6']),
            ('relu3',alexnet.features._modules['7']),
            ('conv4',alexnet.features._modules['8']),
            ('relu4',alexnet.features._modules['9']),
            ('conv5',alexnet.features._modules['10']),
            ('relu5',alexnet.features._modules['11']),
            ('pool5',alexnet.features._modules['12']),
            ('flatten',Flatten()),
            ('fc1',nn.Linear(256 * 2 * 2,103)),
            ('smax' , nn.Softmax(dim=1))
        ]))
        self.smax=nn.Softmax(dim=1)
        
    def forward(self,x):
        x=self.SequentialModel1(x)
        return(x.view(-1,103))
    
model=Transfer()
optimizer = optim.SGD(model.parameters(), lr=0.001,momentum=0.6)
criterion = nn.CrossEntropyLoss()
epochRange=1000
miniBatches=15
batchSize=1000

print("Cell Execution Complete")

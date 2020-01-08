# Elevator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import collections


class elevatorLearning(nn.Module):
    def __init__(self):
        super(elevatorLearning, self).__init__()
        self.model1=nn.Sequential(
            nn.Linear(16,8),
            nn.BatchNorm1d(8),
            nn.ReLU(True),
            nn.Linear(8,4),
            nn.BatchNorm1d(4),
            nn.ReLU(True),
            nn.Linear(4,3),
            nn.Softmax()
        )
    
    def forward(self,x):
        x=self.model1(x)
        return(x)
    
print("The elevator class has been created")

# IMPORTS
import random
import numpy as np
import copy

class ReplayMemory:
    def __init__(self, size):
        self.counter=0
        self.memory = collections.deque(maxlen=size)

    def append(self, element):
        self.memory.append(element)
        self.counter += 1

    def sample(self, n):
        return random.sample(self.memory, n)
    
    def __len__(self):
        return len(self.memory)
    
class elevator(object):
    def __init__(self,numFloors):
        # Current Loc + Outside Buttons Pressed Up + Outside Buttons Pressed Down + Inside Button Pressed + Direction
        self.state=np.array([1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0])
        self.playHistory=ReplayMemory(300)
        self.boardStateDict={}
        self.totalCount=0
        self.RMatrix={}
        self.QMatrix={}
        self.QGamma=0.2
        self.modelLearningRate=0.1
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=self.modelLearningRate, weight_decay=1e-5)
        
    def resetElevator(self):
        self.state=np.array([1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0])
    
    def evaluateState(self):
        # Inside Direction
        if(self.state[20]==1):
            direction='up'
        elif(self.state[21]==1):
            direction='down'
        insides=np.where(self.state[15:20]==1)
        currentState=np.where(self.state[0:5]==1)
        loss=0
        if(direction=='up'):
            loss=loss-np.sum(currentState-insides)
        elif(direction=='down'):
            loss=loss+np.sum(currentState-insides)
        # Outside Buttons
        outsideups=np.where(self.state[5:10]==1)
        outsidedowns=np.where(self.state[10:15]==1)
        if(direction=='up'):
            loss=loss-np.sum(currentState-outsideups)
            loss=loss-np.sum(currentState-outsidedowns)
        elif(direction=='down'):
            loss=loss+np.sum(currentState-outsideups)
            loss=loss+np.sum(currentState-outsidedowns)
            
    def changeState(self,direction):
        if(direction=='up'):
            self.state[20]=1
            self.state[21]=0
            currentState=np.where(self.state[0:5]==1)
            self.state[0:5]=0
            self.state[currentState+1]=1
            self.
        else:
            self.state[20]=0
            self.state[21]=1
        
        
        
        
        

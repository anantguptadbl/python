# Tic Tac Toe Learning with Deep Q Reinforcement Learning ( Well Written)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
import collections

# Pytorch TIC TAC TOE CLASS
# We are going to predict out of the 9 possible spots, where should we place our marker
class tictactoeLearning(nn.Module):
    def __init__(self):
        super(tictactoeLearning, self).__init__()
        self.model1=nn.Sequential(
            nn.Linear(9,5),
            nn.ReLU(True),
            nn.Linear(5,9)
        )
    
    def forward(self,x):
        x=self.model1(x)
        return x
    
print("The tic tac toe learning class has been created")

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

# First we will write a simple TIC TAC TOE cube algo
class tictactoe(object):
    # Constructor
    def __init__(self):
        # We will set the initial board with everything set to 0
        self.board=np.array([0,0,0,0,0,0,0,0,0])
        
        # Creating a board history of 300 moves
        self.playHistory=ReplayMemory(300)
        
        self.curPlayer=2
        self.boardStateDict={}
        self.totalCount=0
        self.RMatrix={}
        self.QMatrix={}
        self.QGamma=0.2
        self.modelLearningRate=0.1
        self.trainCount=0
        self.X=[]
        self.Y=[]
        self.model = tictactoeLearning()
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=self.modelLearningRate, weight_decay=1e-5)
    
    def resetBoard(self):
        self.board=np.array([0.,0.,0.,0.,0.,0.,0.,0.,0.])
        self.curPlayer=2.
        self.curEvaluation=''
    
    def evaluateRows(self,x):
        for curRow in range(3):
            if(np.sum(self.board[curRow*3:(curRow*3)+3]==x)==3):
                return True
        return False
    
    def evaluateColumns(self,x):
        for curCol in range(3):
            if(np.sum(self.board[[curCol,3+curCol,6+curCol]]==x)==3):
                return True
        return False        
    
    def evaluateDiag(self,x):
        if(np.sum(self.board[[0,4,8]]==x)==3):
            return True
        elif(np.sum(self.board[[2,4,6]]==x)==3):
            return True
        else:
            return False
    
    def evaluateDraw(self):
        if(np.sum(self.board==0)==0):
            return True
        else:
            return False
    
    def evaluateBoard(self):
        # This denotes that Player 2 has won
        if((self.evaluateRows(2)==True or self.evaluateColumns(2)==True or self.evaluateDiag(2)==True) and self.curPlayer==2 ):
            return 100
        # This denotes a draw which invokes a lesser positive result w.r.t a pure win
        elif self.evaluateDraw()==True:
            return 50
        # This denotes that Player 2 has lost
        elif ((self.evaluateRows(1)==True or self.evaluateColumns(1)==True or self.evaluateDiag(1)==True) and self.curPlayer==1 ):
            return -100
        else:
            return 0
        
    def evaluateWinner(self):
        # This denotes that Player 2 has won
        if((self.evaluateRows(2)==True or self.evaluateColumns(2)==True or self.evaluateDiag(2)==True) ):
            return 2
        # This denotes a draw which invokes a lesser positive result w.r.t a pure win
        elif self.evaluateDraw()==True:
            return 0
        # This denotes that Player 2 has lost
        elif ((self.evaluateRows(1)==True or self.evaluateColumns(1)==True or self.evaluateDiag(1)==True) ):
            return 1
        else:
            return -1
        
    def toggle(self):
        if(self.curPlayer==1.):
            self.curPlayer=2.
        else:
            self.curPlayer=1.
            
    def setBoardVal(self,player,position):
        self.curPlayer=player
        self.board[position]=float(player)
        self.curBoardEvaluation=self.evaluateBoard()
        if((self.curBoardEvaluation==100) or (self.curBoardEvaluation==50) or (self.curBoardEvaluation==-100)):
            return True
        else:
            return False
    
    def playStepRandom(self):
        self.toggle()
        emptySteps=np.where(self.board==0)[0]
        self.board[random.choice(emptySteps)]=self.curPlayer
        self.curBoardEvaluation=self.evaluateBoard()
        #print("The curBoardEvaluation value is {0}".format(self.curBoardEvaluation))
        # Append the current board state
        #self.boardHistory.append(copy.deepcopy(self.board))
        # Evaluate the board
        #self.curEvaluation=self.evaluateBoard(self.curPlayer)
        if((self.curBoardEvaluation==100) or (self.curBoardEvaluation==50) or (self.curBoardEvaluation==-100)):
            return True
        else:
            return False
        
    def playStepRandomPlayer(self,player):
        self.curPlayer=player
        emptySteps=np.where(self.board==0)[0]
        self.board[random.choice(emptySteps)]=player
        self.curBoardEvaluation=self.evaluateBoard()
        #print("The curBoardEvaluation value is {0}".format(self.curBoardEvaluation))
        # Append the current board state
        #self.boardHistory.append(copy.deepcopy(self.board))
        # Evaluate the board
        #self.curEvaluation=self.evaluateBoard(self.curPlayer)
        if((self.curBoardEvaluation==100) or (self.curBoardEvaluation==50) or (self.curBoardEvaluation==-100)):
            return True
        else:
            return False
    
    def getInitialBatchPredictionsFRomModel(self,batch_size):
        self.resetBoard()
        def executeTraining():
            #if(len(self.playHistory) < batch_size):
            #    curBatch=np.random.choice(self.playHistory,batch_size)
            #else:
            #    curBatch=self.playHistory.sample(batch_size)
            #print(curBatch)
            if(len(self.playHistory) >= batch_size):
                # Take sample from the history
                curBatch=self.playHistory.sample(batch_size)
                # We will now train the batch
                batch_data=np.array([x['state'] for x in curBatch])
                nextStates=self.model(Variable(torch.from_numpy(batch_data.astype(np.float32)))).data.numpy()
                #print("Next States are {0}".format(nextStates))

                # For each row in the batch, where we have a final decision then we will set the nextStates to 0
                for i,curBatchElement in enumerate(curBatch):
                    #print("The curBatchElement reward is {0}".format(curBatchElement['reward']))
                    if(curBatchElement['reward'] in [100,-100,50]):
                        nextStates[i]=np.array([0,0,0,0,0,0,0,0,0])
                        #print("Adjusted the next state")

                # We will now train the model
                dataOutput = self.model(Variable(torch.from_numpy(batch_data.astype(np.float32))))
                #print("The pedicted is {0} and the actual is {1}".format(dataOutput,nextStates))
                loss = self.criterion(dataOutput, Variable(torch.from_numpy(nextStates.astype(np.float32))))
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                return loss
                
        while(1):
            curState=self.board
            curResult=self.playStepRandom()
            #print("We are executing the next step when the board config is {0}".format(self.board))
            self.totalCount=self.totalCount+1
            nextState=self.model(Variable(torch.from_numpy(self.board.astype(np.float32)))).data.numpy()
            if(len(nextState[np.where(curState==0.)]) >0):
                possibleStates=np.where(curState==0.)[0]
                action=possibleStates[np.argmax(nextState[possibleStates])]
                #print("On the state {0} the action is {1} because of {2}".format(self.board,action,np.where(curState==0.)))
                self.playHistory.append({'state':copy.deepcopy(curState),'nextState':nextState,'reward':copy.deepcopy(self.curBoardEvaluation)})
                self.board[action]==2
            if self.totalCount % batch_size == 0:
                curLoss=executeTraining()
            if(curResult == True):
                break
        # After the exit, we will again execute the training
        curLoss=executeTraining()
        return(curLoss)
        
    def playGame(self):
        self.resetBoard()
        #print("Playing the game")
        while(1):
            nextStates=self.model(Variable(torch.from_numpy(self.board.astype(np.float32)))).data.numpy()
            possibleStates=np.where(self.board==0.)[0]
            event=possibleStates[np.argmax(nextStates[possibleStates])]
            if(self.setBoardVal(2.,event)==True):
                break
            #print(self.board)
            self.toggle()
            if(self.playStepRandomPlayer(1.)==True):
                break
            #print(self.board)
            self.toggle()
        #print("The final board is {0}".format(self.board))
        return(self.evaluateWinner())
        
                          
if __name__=="__main__":
    # Running the code
    curBoard=tictactoe()
    for epoch in range(100000):
        loss=curBoard.getInitialBatchPredictionsFRomModel(10)
        if(epoch % 5000==0):
            print("Loss at epoch {0} is {1}".format(epoch,loss))
            # We will also analyse the accuracy of the game
            print("Playing several games and analysing accuracy")
            winBy2=0
            winBy1=0
            draws=0
            for testEpoch in range(100):
                status=curBoard.playGame()
                if(status==2):
                    winBy2=winBy2+1
                if(status==1):
                    winBy1=winBy1+1
                if(status==0):
                    draws=draws+1    
            print("Final stats for epoch {3} are 2Wins:{0}   1Wins:{1}  Draws:{2}".format(winBy2,winBy1,draws,epoch))
            
    # After the training is complete, we will now play the game
    #curBoard.playGame()
    
The tic tac toe learning class has been created
Loss at epoch 0 is None
Playing several games and analysing accuracy
Final stats for epoch 0 are 2Wins:37   1Wins:40  Draws:23
Loss at epoch 5000 is 0.0
Playing several games and analysing accuracy
Final stats for epoch 5000 are 2Wins:55   1Wins:34  Draws:11
Loss at epoch 10000 is 6.20125047134934e-06
Playing several games and analysing accuracy
Final stats for epoch 10000 are 2Wins:70   1Wins:21  Draws:9
Loss at epoch 15000 is 2.3890817146821064e-07
Playing several games and analysing accuracy
Final stats for epoch 15000 are 2Wins:63   1Wins:18  Draws:19
Loss at epoch 20000 is 0.0
Playing several games and analysing accuracy
Final stats for epoch 20000 are 2Wins:60   1Wins:16  Draws:24
Loss at epoch 25000 is 4.639098551706411e-05
Playing several games and analysing accuracy
Final stats for epoch 25000 are 2Wins:37   1Wins:43  Draws:20
Loss at epoch 30000 is 0.0
Playing several games and analysing accuracy
Final stats for epoch 30000 are 2Wins:49   1Wins:41  Draws:10
Loss at epoch 35000 is 1.257896428796812e-06
Playing several games and analysing accuracy
Final stats for epoch 35000 are 2Wins:60   1Wins:34  Draws:6
Loss at epoch 40000 is 0.00010712425137171522
Playing several games and analysing accuracy
Final stats for epoch 40000 are 2Wins:41   1Wins:50  Draws:9
Loss at epoch 45000 is 0.0
Playing several games and analysing accuracy
Final stats for epoch 45000 are 2Wins:68   1Wins:17  Draws:15

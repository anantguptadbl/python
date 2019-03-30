# IMPORTS
import random
import numpy as np
import copy

# Pytorch neural network class



# First we will write a simple TIC TAC TOE cube algo
class tictactoe(object):
    # Constructor
    def __init__(self):
        # We will set the initial board
        self.board=np.array([0,0,0,0,0,0,0,0,0])
        self.curPlayer=2
        self.boardLogs=[]
        self.boardStateDict={}
        self.totalCount=0
        self.RMatrix={}
        self.QMatrix={}
        self.QGamma=0.2
        self.X=[]
        self.Y=[]
    
    def resetBoard(self):
        self.board=np.array([0,0,0,0,0,0,0,0,0])
        self.curPlayer=2
        self.boardLogs=[]
        self.curEvaluation=''
    
    def evaluateRows(self,x):
        #print("Evaluated Rows function {}".format(self.board))
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
    
    def evaluateBoard(self,x):
        # This will tell you the following
        # 1. Game left
        # 2. Draw
        # 3. 1 Winner
        # 4. 2 Winner
        #print("Evaluated board function {}".format(self.board))
        if(self.evaluateRows(x)==True or self.evaluateColumns(x)==True or self.evaluateDiag(x)==True):
            return "winner"
        elif self.evaluateDraw()==True:
            return "draw"
        else:
            return "game left"
        
    def toggle(self):
        if(self.curPlayer==1):
            self.curPlayer=2
        else:
            self.curPlayer=1
    
   
    def playStepRandom(self):
        self.toggle()
        emptySteps=np.where(self.board==0)[0]
        self.board[random.choice(emptySteps)]=self.curPlayer
        # Append the current board state
        self.boardLogs.append(copy.deepcopy(self.board))
        # Evaluate the board
        self.curEvaluation=self.evaluateBoard(self.curPlayer)
        if((self.curEvaluation=="winner") or (self.curEvaluation=="draw")):
            return True
        else:
            return False
    
    def updateQMatrixStep(self):
        self.toggle()
        curState=list(self.board)
        emptySteps=np.where(self.board==0)[0]
        curRandomChoice=random.choice(emptySteps)
        self.board[random.choice(emptySteps)]=self.curPlayer
        nextState=list(self.board)
        self.X.append(curState + nextState)
        emptySteps=np.where(self.board==0)[0]
        allPossibleNextNextStates=[self.returnReplacedArray(self.board,x,self.curPlayer) for x in emptySteps]
        allPossibleQValues=[]
        finalValue=0
        for curNextState in allPossibleNextNextStates:
            if(tuple(nextState+ curNextState) in self.QMatrix):
                allPossibleQValues.append(self.QMatrix[tuple(nextState + curNextState)])
        if(tuple(curState + nextState) in self.RMatrix):
            finalValue=finalValue + self.RMatrix[tuple(curState + nextState)]
        if(len(allPossibleQValues) > 0):
            finalValue=finalValue + max(allPossibleQValues) * self.QGamma
        #if(tuple(curState + nextState) in self.QMatrix):
        #    self.QMatrix[tuple(curState + nextState)]=self.QMatrix[tuple(curState + nextState)] + finalValue
        #else:
        self.QMatrix[tuple(curState + nextState)]=finalValue
        self.Y.append(self.QMatrix[tuple(curState + nextState)])
        # Evaluate the board
        self.curEvaluation=self.evaluateBoard(self.curPlayer)
        if((self.curEvaluation=="winner") or (self.curEvaluation=="draw")):
            return True
        else:
            return False
        
        #Q(state, action) = R(state, action) + Gamma * Max[Q(next state, all actions)]
        #Q(1, 5) = R(1, 5) + 0.8 * Max[Q(5, 1), Q(5, 4), Q(5, 5)] = 100 + 0.8 * 0 = 100
    
    def returnReplacedArray(self,arr,position,value):
        arr1=copy.deepcopy(arr)
        arr1[position]=value
        return arr1
    
    # Q Learning : Q Matrix
    def createQMatrix(self):
        while(self.updateQMatrixStep() != True):
            a=1
        print("We have the Q matrix data for the neural network training")
        print(self.X)
        print(self.Y)
        print("We will now be passing this data to the training module")
        
            
    # Q Learning : R Matrix   
    def createRMatrix(self):
        while(self.playStepRandom() != True):
            a=1
        # We will now add the second last step in the transition matrix
        endState=list(self.boardLogs[-1])
        priorEndState=list(self.boardLogs[-2])
        if( tuple(priorEndState + endState) in self.RMatrix):
            a=1
        else:                   
            if((self.curEvaluation=="winner") and (self.curPlayer==1)):
                self.RMatrix[tuple(priorEndState + endState)]=100
            elif((self.curEvaluation=="winner") and (self.curPlayer==2)):
                self.RMatrix[tuple(priorEndState + endState)]=-100
            elif(self.curEvaluation=="draw"):
                self.RMatrix[tuple(priorEndState + endState)]=50
        # Appending the total count
        #self.totalCount=self.totalCount+1
                    
    def predictNextQState(self,x):
        # Here we will be using the Q matrix to come up with the next state
        curState=self.board
        emptySteps=np.where(self.board==0)[0]
        nextCombinations=[]
        for nextStep in emptySteps:
            origArray=copy.deepcopy(self.board)
            origArray[nextStep]=self.curPlayer
            if(tuple(list(curState) + list(origArray)) in self.QMatrix):
                nextCombinations.append([origArray,self.QMatrix[tuple(list(curState) + list(origArray))]])
            else:
                nextCombinations.append([origArray,0])
        # Of all the possible combinations, we are choosing the one with the maximum return
        nextCombinations.sort(key=lambda x : x[1],reverse=True)
        nextStep=nextCombinations[0][0]
        for curStep in emptySteps:
            if(nextStep[curStep] <> '0'):
                return curStep
        
    def playGame(self):
        while(self.playStep() != True):
            print("the current state of the board is {}".format(self.board))
            a=1

    def playStep(self):
        self.toggle()
        emptySteps=np.where(self.board==0)[0]
        print("the current state of the board is {}".format(self.board))
        if(self.curPlayer==2):
            randomChoice=random.choice(emptySteps)
            print("The random choice made is {}".format(randomChoice))
            self.board[random.choice(emptySteps)]=self.curPlayer
            self.curEvaluation=self.evaluateBoard(self.curPlayer)
            # Evaluate the board
            if(self.curEvaluation=="winner"):
                print("The winner is {}".format(self.curPlayer))
                return True
            if(self.curEvaluation=="draw"):
                print("The game has been drawn")
                return True
            else:
                return False
        else:
            stepLocation=self.predictNextQState(self.board)
            self.board[stepLocation]=self.curPlayer
            # Evaluate the board
            self.curEvaluation=self.evaluateBoard(self.curPlayer)
            if(self.curEvaluation=="winner"):
                print("The winner is {}".format(self.curPlayer))
                return True
            if(self.curEvaluation=="draw"):
                print("The game has been drawn")
                return True
            else:
                return False
           
if __name__=="__main__":
    # Running the code
    curBoard=tictactoe()
    curBoard.createRMatrix()
    for x in range(1000):
        curBoard.resetBoard()
        curBoard.createRMatrix()
    print("We now have the R Matrix")
    for x in range(5):
        curBoard.resetBoard()
        curBoard.createQMatrix()
    print("We now have the Q Matrix")
    # We will now play the game
    #curBoard.resetBoard()
    #curBoard.playGame()

# IMPORTS
import random
import numpy as np
import copy

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
        self.boardLogs.append(self.board)
        # Evaluate the board
        self.curEvaluation=self.evaluateBoard(self.curPlayer)
        if((self.curEvaluation=="winner") or (self.curEvaluation=="draw")):
            return True
        else:
            return False
        
    def playGameTraining(self):
        while(self.playStepRandom() != True):
            a=1
        #if(self.curEvaluation=="winner"):
        #    print("{} won".format(self.curPlayer))
        #else:
        #    print("Match Drawn")
        self.totalCount=self.totalCount+1
        self.refreshProbRatios()
    
    def refreshProbRatios(self):
        if(self.curEvaluation=="winner" and self.curPlayer==1):
            for x in self.boardLogs:
                curKey=''.join(np.array(x,dtype='str'))
                if(curKey in self.boardStateDict):
                    self.boardStateDict[curKey]=self.boardStateDict[curKey] + 1
                else:
                    self.boardStateDict[curKey]=1
                    
    def predictNextState(self,x):
        # Now using the state probabilities of winning, we will now predict the next step
        emptySteps=np.where(self.board==0)[0]
        nextCombinations=[]
        for nextStep in emptySteps:
            for numberCombinations in [1,2]:
                origArray=copy.deepcopy(self.board)
                origArray[nextStep]=numberCombinations
                curKey=''.join(np.array(origArray,dtype='str'))
                if(curKey not in self.boardStateDict):
                    nextCombinations.append([curKey,0.0])
                else:
                    nextCombinations.append([curKey,self.boardStateDict[curKey]])
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
            self.board[random.choice(emptySteps)]=self.curPlayer
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
            stepLocation=self.predictNextState(self.board)
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
    curBoard.playGameTraining()
    for x in range(10000):
        curBoard.resetBoard()
        curBoard.playGameTraining()
    print("The games have been played")
    for x in curBoard.boardStateDict:
        curBoard.boardStateDict[x]=(1.00 * curBoard.boardStateDict[x]) / curBoard.totalCount

    # We will now play the actual game with predictions coming from trained data
    curBoard.resetBoard()
    curBoard.playGame()

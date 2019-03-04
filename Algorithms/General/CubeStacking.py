# Imports
import numpy as np

def getAllCombinations(curState):
    combinationList=[]
    combinations={}
    for x in range(len(curState)):
        combinationList.append(tuple(np.roll(curState,x)))
        combinationList.append(tuple(np.roll(curState,x)[::-1]))
    combinationList=set(combinationList)
    for s1,s2,s3,s4 in combinationList:
        combinations.setdefault(s1, {}).setdefault(s2, {})[s3] = s4
    return(combinations)
        
def generateCombo(x,allComboDict,status1,status2,status3,status4):
    #print("Entered generate combo for x :{} statusDict : {}".format(x,statusDict))
    for s1 in allComboDict[x]:
        if(s1 in status1):
            continue
        for s2 in allComboDict[x][s1]:
            if(s2 in status2):
                continue
            for s3 in allComboDict[x][s1][s2]:
                if(s3 in status3):
                    continue
                if(allComboDict[x][s1][s2][s3] in status4):
                    continue
                else:
                    s4=allComboDict[x][s1][s2][s3]
                    yield([s1,s2,s3,s4])

def getValidNextLayer(curCubeLayer,allComboDict,status1,status2,status3,status4):
    sizeVals=[]
    #print("Entered getValidNextLayer curLayer : {}, status1 {} status2 {} status3 {} status4 {}".format(curCubeLayer,status1,status2,status3,status4))
    if(curCubeLayer >= numCubeLayers):
        #print("Exiting as reached maximum cube Layers")
        return 0
    cubeLayerLists=list(generateCombo(curCubeLayer,allComboDict,status1,status2,status3,status4))
    #print("For curCubeLayer {}, the list size is {}".format(curCubeLayer,len(cubeLayerLists)))
    if(len(cubeLayerLists)==0):
        return 0
    else:
        return(max([
            1 + 
             getValidNextLayer
             (
                 curCubeLayer+1,
                 allComboDict,
                 status1 + [s1],
                 status2 + [s2],
                 status3 + [s3],
                 status4 + [s4]
             )
            for s1,s2,s3,s4 in cubeLayerLists
        ]))

# Get All Combinations
def getAllCombinationsDict(numCubeLayers,origArray):
    allComboDict={}
    for x in range(numCubeLayers):
        allComboDict[x]=getAllCombinations(origArray[x])
    return(allComboDict)

# Initialization
# TEST 1
#origArray=np.array([[1,2,3,4],[2,3,4,5],[3,4,5,1],[4,5,1,2],[5,1,2,3]])
# TEST 2
#origArray=np.array([[1,2,3,4],[1,2,3,4],[3,4,5,1],[4,5,1,2],[5,1,2,3]])
# TEST 3
#origArray=np.array([[1,2,3,4],[1,2,3,4],[1,2,3,4],[4,5,1,2],[5,1,2,3]])
# TEST 4
#origArray=np.dstack((
#    np.random.randint(1,10,size=(100000)),
#    np.random.randint(1,10,size=(100000)),
#    np.random.randint(1,10,size=(100000)),
#    np.random.randint(1,10,size=(100000))
#))[0]
# TEST 5
#origArray=np.array([[1,2,3,4],[2,3,4,5],[3,4,5,6],[4,5,6,7],[5,6,7,8],[7,8,9,10],[9,10,11,12],[10,11,12,13],[11,12,13,14]])
# TEST 6
origArray=np.array([[1,2,3,4],[2,3,4,5],[3,4,5,6],[4,5,6,7],[5,6,7,8],[7,8,9,10],[9,10,11,12],[1,1,1,1],[1,1,1,1]])

# Init Data Stats
print(origArray.shape)
numCubeLayers=origArray.shape[0]
print("Number of stacked layers are {}".format(numCubeLayers))
# For each cube store all the possible combinations

allComboDict=getAllCombinationsDict(numCubeLayers,origArray)
print("The maximum height is {}".format(getValidNextLayer(0,allComboDict,[],[],[],[])))

import numpy as np
import pandas as pd
import math

a=np.array([[0,0,1],[0,1,1],[1,2,1],[2,3,1],[2,2,1],[3,3,1],[3,1,0],[4,1,0],[4,0,1]])


# Assumptions
# 1) User should be in incremental order starting from 0
# 2) Item should be in incremental order starting from 0

batchSize=3
numEpochs=1000
# Init Parameters
latentFeatures=2
momentum=0.8
epsilon=1
regLambda=0.1
meanLabels=np.mean(a[:, 2]) 
print("Mean is {}".format(meanLabels))
totalLength=a.shape[0]
#Number of Users
numUsers=max(a[:,0]) + 1
userArray=a[:,0]
numItems=max(a[:,1]) + 1
itemArray=a[:,1]

# Initialization
# This is the User Embedding. Initialized with random values, but with each iteration and epoch it will arrive towards the true embeddings
U=np.random.rand(numUsers,latentFeatures)
# This is the Item Embedding. Initialized with random values, but with each iteration and epoch it will arrive towards the true embeddings
V=np.random.rand(numItems,latentFeatures)
# This is the diff 
dUDelta=np.zeros((numUsers,latentFeatures))
dVDelta=np.zeros((numItems,latentFeatures))
batchLength=int(math.ceil(totalLength / batchSize))

# Each Epoch will work on all possible combinations of a chunk square matrix of user and items
for curEpoch in range(numEpochs):
    # Each Batch will iterate over the entire data set once
    # The iteration will be randomized in terms of selecting the square chunk. They will not necessarily be in order
    for curBatch in range(batchLength-1):
        # Get the random batch
        randomizedRows=np.arange(totalLength)
        np.random.shuffle(randomizedRows)

        # Current Batch. This will get the users and items that will be used for calculations
        batchRange=np.arange(batchSize * curBatch,batchSize * (curBatch +1))
        # The following multiplication and sum eventually 
        prediction=np.sum(np.multiply(U[userArray[randomizedRows[batchRange]],:],V[itemArray[randomizedRows[batchRange]],:]),axis=1)
        actual=a[randomizedRows[batchRange],2]
        error= prediction - actual + meanLabels
        print("Error is {}".format(error))
        # Gradient Approximation
        dU=2 * np.multiply(error[:,np.newaxis],V[itemArray[randomizedRows[batchRange]],:]) + regLambda * U[userArray[randomizedRows[batchRange]],:]
        dV=2 * np.multiply(error[:,np.newaxis],U[userArray[randomizedRows[batchRange]],:]) + regLambda * V[itemArray[randomizedRows[batchRange]],:]

        # Aggregation of gradients at the user level
        dU_Agg = np.zeros((numUsers, latentFeatures))
        dV_Agg = np.zeros((numItems, latentFeatures))
        # Adding the aggregations at the user and items
        for i in range(batchSize):
            dU_Agg[userArray[randomizedRows[i]],:]+=dU[i,:]
            dV_Agg[itemArray[randomizedRows[i]],:]+=dV[i,:]
        # Weight update factor
        dUDelta=momentum * dUDelta + (epsilon * dU_Agg / batchSize)
        dVDelta=momentum * dVDelta + (epsilon * dV_Agg / batchSize)

        U=U-dUDelta
        V=V-dVDelta

    # After all batches have been covered, the final cost function
    prediction=np.sum(np.multiply(U[userArray[randomizedRows[batchRange]],:],V[itemArray[randomizedRows[batchRange]],:]),axis=1)
    actual=a[randomizedRows[batchRange],2]
    error= prediction - actual + meanLabels
    obj = np.linalg.norm(error) ** 2 + 0.5 * regLambda * (np.linalg.norm(dU) ** 2 + np.linalg.norm(dV) ** 2)
    if(curEpoch % 100 ==0):
        print("For epoch {} the objective function value is {}".format(curEpoch,np.sqrt(obj/a.shape[0])))

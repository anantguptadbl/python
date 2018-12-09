# Assumptions
# 1) User should be in incremental order starting from 0
# 2) Item should be in incremental order starting from 0

# Init Parameters
latentFeatures=2
meanLabels=np.mean(a[:, 2]) 
print("Mean is {}".format(meanLabels))
# Get the random batch
totalLength=a.shape[0]
randomizedRows=np.arange(totalLength)
np.random.shuffle(randomizedRows)

#Number of Users
numUsers=max(a[:,0]) + 1
userArray=a[:,0]
numItems=max(a[:,1]) + 1
itemArray=a[:,1]

# Initialization
U=np.random.rand(numUsers,latentFeatures)
print(U.shape)
V=np.random.rand(numItems,latentFeatures)
print(V.shape)

# Current Batch
batchRange=[0,1]
print(randomizedRows)
print(randomizedRows[batchRange])
print(U[userArray[randomizedRows[batchRange]],:])
print(V[itemArray[randomizedRows[batchRange]],:])
prediction=np.multiply(U[userArray[randomizedRows[batchRange]],:],V[itemArray[randomizedRows[batchRange]],:])
actual=a[randomizedRows[batchRange],2]
error= prediction - actual + meanLabels
print(error)

# FFM
import numpy as np

learningRate=0.001
regParam=0.01
a=np.array([[5,2,0,0,-2],[5,2,2,0,-2],[-2,2,2,0,0]])

# For Field Aware Machines we need the categories for each user and each item combination
userCategories=np.array(['userType1','userType1','userType2'])
itemCategories=np.array(['itemType1','itemType2','itemType2','itemType2','itemType3'])

userCategoriesDict=dict((x,i) for i,x in enumerate(np.unique(userCategories)))
itemCategoriesDict=dict((x,i) for i,x in enumerate(np.unique(itemCategories)))

x=np.random.rand(len(np.unique(itemCategories)),3,2)
y=np.random.rand(len(np.unique(userCategories)),5,2)
biasx=np.zeros(3)
biasy=np.zeros(5)
numElements=a.shape[0] * a.shape[1]

# We will take only those elements, which have nonZero data in them
loopValues=[(i,j,a[i][j]) for i in range(a.shape[0]) for j in range(a.shape[1]) if a[i][j] <> 0]

loopCounter=0
while(1):
    loopCounter+=1
    totError=0
    for i,j,scoreVal in loopValues:
        curPrediction=np.matmul(x[itemCategoriesDict[itemCategories[j]],i,:],y[userCategoriesDict[userCategories[i]],j,:].T)
        curError=a[i][j] - curPrediction
        totError+=curError 
        x[itemCategoriesDict[itemCategories[j]],i,:]=  x[itemCategoriesDict[itemCategories[j]],i,:] + learningRate * (curError*y[userCategoriesDict[userCategories[i]],j,:] - regParam * x[itemCategoriesDict[itemCategories[j]],i,:])
        y[userCategoriesDict[userCategories[i]],j,:]=  y[userCategoriesDict[userCategories[i]],j,:] + learningRate * (curError*x[itemCategoriesDict[itemCategories[j]],i,:] - regParam * y[userCategoriesDict[userCategories[i]],j,:])
    if((1.000*totError)/numElements < 0.01):
        print("Achieved in {} iterations".format(loopCounter))
        break
        
            

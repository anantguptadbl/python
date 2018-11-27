# Restricted Boltzmann Machines

import numpy as np
import pandas as pd

# Helper Functions
def logisticFunc(x):
    return 1.0 / (1 + np.exp(-x))

inpArray=np.array([[1,0,0,0,0,1,0],[0,1,1,1,1,0,0],[1,1,1,0,0,0,0],[0,0,1,0,0,1,1],[1,1,1,1,1,0,0],[0,0,0,0,0,0,1]])
hiddenFactors=3
weights=np.random.rand(inpArray.shape[1],hiddenFactors)
inpBias=np.random.rand(inpArray.shape[1])
hiddenBias=np.random.rand(hiddenFactors)
learningRate=0.01
gibbsSamplingStep=5
epochs=1000

initArray=inpArray

for curEpoch in range(epochs):
    # Gibbs Sampling Step
    for curStep in range(gibbsSamplingStep):
        firstHiddenPass=np.dot(initArray,weights)
        firstHiddenPassProbs=logisticFunc(firstHiddenPass)
        firstPassPositiveActivations=np.dot(inpArray.T,firstHiddenPassProbs)

        firstRecreated=np.dot(firstHiddenPassProbs,weights.T)
        secondHiddenPass=np.dot(firstRecreated,weights)
        secondHiddenPassProbs=logisticFunc(secondHiddenPass)
        secondPassPositiveActivations=np.dot(firstRecreated.T,secondHiddenPassProbs)

        # Metrics
        error=np.sum((inpArray - firstRecreated ) ** 2)
        energy=np.sum(np.dot(firstHiddenPassProbs.T,np.dot(inpArray,weights)))
        print("epoch : {} gibbs step : {}  Loss :{} Energy :{}".format(curEpoch,curStep,error,energy))

        # Reset
        initArray=firstRecreated

    # Update Weights after Gibbs Sampling
    weightsUpdate=learningRate * (firstPassPositiveActivations - secondPassPositiveActivations)
    weights=weights + weightsUpdate
    
    # Recommendation : Testing
recArray=np.array([1,1,1,0,0,0,0])

# We also need that which product triggers the hidden factors by what amount
# For each hidden factor, we will get the list of products by increasing order of probability
productFactor=[]
for curHiddenFactor in range(hiddenFactors):
    for curProduct in range(weights.shape[0]):
        productFactor.append([curHiddenFactor,curProduct,weights[curProduct,curHiddenFactor]])
productFactor=pd.DataFrame(productFactor,columns=['hiddenFactor','product','weights'])

# Find that for the test input array, what are the activations of the hidden factors
activations=pd.DataFrame(zip(range(hiddenFactors),logisticFunc(np.dot(recArray,weights))),columns=['hiddenFactor','activations'])

# Join the data with the training product activations at the latent factor level
recs=productFactor.merge(activations,left_on='hiddenFactor',right_on='hiddenFactor',how='inner')
recs['netActivation']=recs.apply(lambda row : row['activations'] * row['weights'],axis=1)

# Get the final results for the test input array
results=pd.DataFrame(zip(recs['product'].unique(),recs.groupby('product')['netActivation'].max(),recs.groupby('product')['netActivation'].min()),columns=['product','highRec','lowRec'])

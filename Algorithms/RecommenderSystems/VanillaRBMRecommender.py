# Restricted Boltzmann Machines

import numpy as np
import pandas as pd

class VanillaRBM():
    def __init__(self,inpArray,hiddenFactors,learningRate,gibbsSamplingStep,epochs):
        self.inpArray=inpArray
        self.hiddenFactors=hiddenFactors
        self.learningRate=learningRate
        self.gibbsSamplingStep=gibbsSamplingStep
        self.epochs=epochs
        self.weights=np.random.rand(inpArray.shape[1],hiddenFactors)
        self.inpBias=np.random.rand(inpArray.shape[1])
        self.hiddenBias=np.random.rand(hiddenFactors)
        self.initArray=self.inpArray
        
    # Helper Functions
    def logisticFunc(x):
        return 1.0 / (1 + np.exp(-x))

    # In this module
    # 1) We will not be using any biases
    # 2) Instead of hard coded 0.5 for the gibbs sampling, we will be doing bernoulli sampling on the hidden probs
    def train2(self):
        for curEpoch in range(self.epochs):
            # Gibbs Sampling Step
            for curStep in range(self.gibbsSamplingStep):
                firstHiddenPass=np.dot(self.initArray,self.weights)
                firstHiddenPassProbs=logisticFunc(firstHiddenPass)
                # The following will apply a binomial probablity to the hidden vectors. This will help us work with neighbourhood
                firstHiddenPassProbsCutOff=np.array([[np.random.binomial(1,i) for i in line] for line in firstHiddenPassProbs])
                firstPassPositiveActivations=np.dot(inpArray.T,firstHiddenPassProbs)

                firstRecreated=np.dot(firstHiddenPassProbsCutOff,self.weights.T)
                secondHiddenPass=np.dot(firstRecreated,self.weights)
                secondHiddenPassProbs=logisticFunc(secondHiddenPass)
                secondPassPositiveActivations=np.dot(firstRecreated.T,secondHiddenPassProbs)

                # Metrics
                error=np.sum((self.inpArray - firstRecreated ) ** 2)
                energy=np.sum(np.dot(firstHiddenPassProbs.T,np.dot(self.inpArray,self.weights)))
                if( (curEpoch%1000==0) & (curStep % self.gibbsSamplingStep==0) ):
                    print("epoch : {} gibbs step : {}  Loss :{} Energy :{}".format(curEpoch,curStep,error,energy))

                # Reset
                self.initArray=firstRecreated

            # Update Weights after Gibbs Sampling
            weightsUpdate=learningRate * ((firstPassPositiveActivations - secondPassPositiveActivations)/self.gibbsSamplingStep)
            # We need to divide by gibbsSamplingStep because, when we use the following formula
            # <vk i hj>data − <vk i hj>model
            # We need the frequency of the times when we have k ( in this case just 1) on together in both data and hidden layer
            # Therefore for every epoch , we will have to consider the number of times each gibbs sampling was able to
            # switch both data and hidden layer ( ON or OFF )
            self.weights=self.weights + weightsUpdate
            
    # In this module
    # 1) We will not be using any biases
    # 2) We will be calcing the error on the original array and not the input array
    def train1(self):
        for curEpoch in range(self.epochs):
            # Gibbs Sampling Step
            for curStep in range(self.gibbsSamplingStep):
                firstHiddenPass=np.dot(self.initArray,self.weights)
                firstHiddenPassProbs=logisticFunc(firstHiddenPass)
                firstPassPositiveActivations=np.dot(inpArray.T,firstHiddenPassProbs)

                firstRecreated=np.dot(firstHiddenPassProbs,self.weights.T)
                secondHiddenPass=np.dot(firstRecreated,self.weights)
                secondHiddenPassProbs=logisticFunc(secondHiddenPass)
                secondPassPositiveActivations=np.dot(firstRecreated.T,secondHiddenPassProbs)

                # Metrics
                error=np.sum((self.inpArray - firstRecreated ) ** 2)
                energy=np.sum(np.dot(firstHiddenPassProbs.T,np.dot(self.inpArray,self.weights)))
                if( (curEpoch%1000==0) & (curStep % self.gibbsSamplingStep==0) ):
                    print("epoch : {} gibbs step : {}  Loss :{} Energy :{}".format(curEpoch,curStep,error,energy))

                # Reset
                self.initArray=firstRecreated

            # Update Weights after Gibbs Sampling
            weightsUpdate=learningRate * (firstPassPositiveActivations - secondPassPositiveActivations)
            self.weights=self.weights + weightsUpdate
            
    # This testing is inline with the train1 module
    def test1(self,recArray):
        # We also need that which product triggers the hidden factors by what amount
        # For each hidden factor, we will get the list of products by increasing order of probability
        productFactor=[]
        for curHiddenFactor in range(self.hiddenFactors):
            for curProduct in range(self.weights.shape[0]):
                productFactor.append([curHiddenFactor,curProduct,self.weights[curProduct,curHiddenFactor]])
        productFactor=pd.DataFrame(productFactor,columns=['hiddenFactor','product','weights'])

        # Find that for the test input array, what are the activations of the hidden factors
        activations=pd.DataFrame(zip(range(self.hiddenFactors),logisticFunc(np.dot(recArray,self.weights))),columns=['hiddenFactor','activations'])

        # Join the data with the training product activations at the latent factor level
        recs=productFactor.merge(activations,left_on='hiddenFactor',right_on='hiddenFactor',how='inner')
        recs['netActivation']=recs.apply(lambda row : row['activations'] * row['weights'],axis=1)

        # Get the final results for the test input array
        results=pd.DataFrame(zip(recs['product'].unique(),recs.groupby('product')['netActivation'].max(),recs.groupby('product')['netActivation'].min()),columns=['product','highRec','lowRec'])
        return(results)

if __name__=="__main__":
    inpArray=np.array([[1,0,0,0,0,1,0],[0,1,1,1,1,0,0],[1,1,1,0,0,0,0],[0,0,1,0,0,1,1],[1,1,1,1,1,0,0],[0,0,0,0,0,0,1]])
    learningRate=0.01
    gibbsSamplingStep=5
    epochs=10000
    hiddenFactors=3
    
    # Initializing and training the RBM
    rbm=VanillaRBM(inpArray,hiddenFactors,learningRate,gibbsSamplingStep,epochs)
    rbm.train1()
    
    # Recommendation : Testing
    recArray=np.array([1,1,1,0,0,0,0])
    print(rbm.test1(recArray))
    
    # Initializing and training the RBM
    rbm=VanillaRBM(inpArray,hiddenFactors,learningRate,gibbsSamplingStep,epochs)
    rbm.train2()
    
    # Recommendation : Testing
    recArray=np.array([1,1,1,0,0,0,0])
    print(rbm.test1(recArray))

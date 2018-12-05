# Based on the following paper
# https://arxiv.org/abs/1507.06803
# A Neighbourhood-Based Stopping Criterion for Contrastive Divergence Learning
# Restricted Boltzmann Machines 
import numpy as np 
import pandas as pd 
import scipy.spatial

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
    def logisticFunc(self,x): 
        return 1.0 / (1 + np.exp(-x)) 
 
     # In this module 
     # 1) We will not be using any biases 
     # 2) We will be calcing the error on the original array and not the input array 
    def train1(self): 
        distances=scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(inpArray,metric='hamming'))
        distances=[list(np.argsort(distances[x])[0:3]) for x in range(distances.shape[0])]
        
        numerator=np.ones(self.initArray.shape[0])
        for curEpoch in range(self.epochs): 
            # Gibbs Sampling Step 
            for curStep in range(self.gibbsSamplingStep): 
                #print("initArray is {}".format(self.initArray))
                #print("weights is {}".format(self.weights))
                firstHiddenPass=np.dot(self.initArray,self.weights) 
                #print("FirstHiddenPass is {}".format(firstHiddenPass))
                firstHiddenPassProbs=self.logisticFunc(firstHiddenPass) 
                firstPassPositiveActivations=np.dot(inpArray.T,firstHiddenPassProbs) 

                firstRecreated=np.dot(firstHiddenPassProbs,self.weights.T) 
                firstRecreatedProbs=self.logisticFunc(firstRecreated)
                secondHiddenPass=np.dot(firstRecreated,self.weights) 
                secondHiddenPassProbs=self.logisticFunc(secondHiddenPass) 
                secondPassPositiveActivations=np.dot(firstRecreated.T,secondHiddenPassProbs) 

                # Metrics 
                error=np.sum((self.inpArray - firstRecreated ) ** 2) 
                energy=np.sum(np.dot(firstHiddenPassProbs.T,np.dot(self.inpArray,self.weights))) 
                if( (curEpoch%10000==0) & (curStep % self.gibbsSamplingStep==0) ): 
                    print("epoch : {} gibbs step : {}  Loss :{} Energy :{}".format(curEpoch,curStep,error,energy)) 

                # Reset 
                self.initArray=firstRecreated 
                # Get the recurring product of Probabilities at each gibbs sampling
                numerator=numerator * (np.sum(firstHiddenPassProbs,axis=1)/firstHiddenPassProbs.shape[1]) * (np.sum(firstRecreatedProbs,axis=1)/firstRecreatedProbs.shape[1])
                
            # Update Weights after Gibbs Sampling 
            weightsUpdate=learningRate * (firstPassPositiveActivations - secondPassPositiveActivations) 
            self.weights=self.weights + weightsUpdate 
            
            # We will introduce the data as per  Neighbourhood based stopping criterion
            # Step 1: First we will get the product of probabilities for all the rows
            numerator=[ x** (1.0000/self.gibbsSamplingStep) for x in numerator]
            #numerator=np.product(np.sum(firstHiddenPassProbs,axis=1))
            # Step 2 : We will find the sum of Probability of the points which are at hamming distance ( closest 3 )
            denominator=[]
            hiddenProbs=np.sum(firstHiddenPassProbs,axis=1)
            recreatedProbs=np.sum(firstRecreated,axis=1)
            #print("Hidden Probs are {}".format(hiddenProbs))
            #print("Recreated Probs are {}".format(recreatedProbs))
            for x in distances:
                #denominator.append(sum([hiddenProbs[y] for y in x]) * sum([recreatedProbs[y] for y in x]))
                denominator.append(sum([hiddenProbs[y] * recreatedProbs[y] for y in x])/3)
            #print("Denominator is {}".format(denominator))
            #print("Numerator is {}".format(numerator))
            criterion=np.prod([numerator[x]/denominator[x] for x in range(len(numerator))])
            if(curEpoch % 10000==0):
                print(criterion)
                    
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
        activations=pd.DataFrame(zip(range(self.hiddenFactors),self.logisticFunc(np.dot(recArray,self.weights))),columns=['hiddenFactor','activations']) 


        # Join the data with the training product activations at the latent factor level 
        recs=productFactor.merge(activations,left_on='hiddenFactor',right_on='hiddenFactor',how='inner') 
        recs['netActivation']=recs.apply(lambda row : row['activations'] * row['weights'],axis=1) 


        # Get the final results for the test input array 
        results=pd.DataFrame(zip(recs['product'].unique(),recs.groupby('product')['netActivation'].max(),recs.groupby('product')['netActivation'].min()),columns=['product','highRec','lowRec']) 
        return(results) 
  

if __name__=="__main__": 
    inpArray=np.array([[1,0,0,0,0,1,0],[0,1,1,1,1,0,0],[1,1,1,0,0,0,0],[0,0,1,0,0,1,1],[1,1,1,1,1,0,0],[0,0,0,0,0,0,1]]) 
    learningRate=0.05 
    gibbsSamplingStep=2
    epochs=100000
    hiddenFactors=3 

    # Initializing and training the RBM 
    rbm=VanillaRBM(inpArray,hiddenFactors,learningRate,gibbsSamplingStep,epochs) 
    rbm.train1() 

    # Recommendation : Testing 
    recArray=np.array([1,1,1,0,0,0,0]) 
    print(rbm.test1(recArray)) 

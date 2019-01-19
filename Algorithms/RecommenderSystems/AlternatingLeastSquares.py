# ALS : Alternating Least Squares Algorithm
import numpy as np

class ALS():

    def __init__(self,inpArray,lambdaVal,latentFeatures):
        # Configuration
        self.numUsers=inpArray.shape[0]
        self.numProducts=inpArray.shape[1]
        self.k=latentFeatures
        self.a=inpArray
        self.lambdaVal=lambdaVal

        # Initialization
        self.x=np.random.rand(self.numUsers,self.k)
        self.y=np.random.rand(self.numProducts,self.k)

    def trainData(self):
        prevError=0
        alteredLambda=0
        while(1):
            # Update the Products keeping the user constant in each loop
            #print("Initial X is {}".format(x))
            for i in range(self.numUsers):
                self.x[i,:]=np.dot(np.linalg.inv(np.dot(self.y.T,self.y) + (self.lambdaVal * np.eye(self.y.shape[1]))),np.dot(self.y.T,a[i].reshape(len(self.a[i]),1))).T 

            # Update the Users keeping the product constant in each loop
            for i in range(self.numProducts):
                self.y[i,:]=np.dot(np.linalg.inv(np.dot(self.x.T,self.x) + (self.lambdaVal * np.eye(self.x.shape[1]))),np.dot(self.x.T,self.a[:,i].reshape(len(self.a[:,i]),1))).T
            curError=sum(sum(self.a-np.dot(self.x,self.y.T)))
            #print("curError is {} and prevError is {} and alteredStatus is {}".format(curError,prevError,alteredLambda))
            if((prevError==curError) and alteredLambda==1):
                break
            if(round(prevError,10)==round(curError,10)):
                alteredLambda=1
                self.lambdaVal=self.lambdaVal * self.lambdaVal
                print("Lambda altered to {}".format(self.lambdaVal))
            else:
                alteredLambda=0
            prevError=sum(sum(self.a-np.dot(self.x,self.y.T)))
    
if __name__=="__main__":
    from scipy.spatial import distance
    a=np.array([[5,2,0,0,-2],[5,2,2,0,-2],[-2,2,2,0,0]])
    als=ALS(a,0.1,2)
    print(distance.squareform(distance.pdist(als.x)))

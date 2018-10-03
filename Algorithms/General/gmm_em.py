# GMM
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class GMM():
    def __init__(self,data):
        self.data=data
        self.cols=self.data.columns.values
        self.data['label']=9999
        self.initialGaussianCenteres=2
        
    def setConfig(self,initialGaussianCenteres):
        self.initialGaussianCenteres=initialGaussianCenteres
        self.config={
            'means' : [[[np.random.choice(self.data[self.cols[i]]) for i in range(len(self.cols))]] * self.initialGaussianCenteres][0],
            'deviations' : [[np.random.randint(10) for j in range(len(self.cols))] for i in range(self.initialGaussianCenteres)],
            'weights' : np.random.dirichlet(np.ones(self.initialGaussianCenteres),size=1)[0]  
        }
    def prob(self,val, mu, sig, probWeight):
        p = probWeight
        #print("Val = {} and mu ={} and sig ={} and probWeight ={}".format(val,mu,sig,probWeight))
        for i in range(len(self.cols)):
            p *= scipy.stats.norm.pdf(val[i], mu[i], sig[i])
        return p

    def expectationStep(self):
        # At this step we will compute the allocation of gaussian centres based on the current data for each point
        self.data['label']=self.data[self.cols].apply(lambda row: np.argmax(self.prob(row.values, self.config['means'][x],self.config['deviations'][x],self.config['weights'][x]) for x in range(self.initialGaussianCenteres)) ,axis=1)        
        #for curRow in range(self.data.shape[0]):
        #    probList=[self.prob(self.data[self.cols].ix[curRow].values, self.config['means'][x],self.config['deviations'][x],self.config['weights'][x]) for x in range(self.initialGaussianCenteres)]
        #    self.data['label'][curRow]=np.argmax(probList)
    
    def maximizationStep(self):
        oldMean=self.config['means']
        self.config['weights']=[
            (self.data[self.data['label']==x].shape[0]*1.0000) / self.data.shape[0] for x in range(self.initialGaussianCenteres)
        ]
        self.config['means']=[
            self.data[self.data['label']==x].mean() for x in range(self.initialGaussianCenteres)
        ]
        
        self.config['means']=[
            self.data[self.data['label']==x][self.cols].mean() for x in range(self.initialGaussianCenteres)
        ]
        self.config['deviations']=[
            self.data[self.data['label']==x][self.cols].std() for x in range(self.initialGaussianCenteres)
        ]
        
        # we will calculate the change in means and return that value
        print(self.config['means'])
        print(oldMean)
        return(sum([np.linalg.norm(self.config['means'][x]-oldMean[x]) for x in range(self.initialGaussianCenteres)]))
    
    def iterate(self):
        change=999
        iterCounter=0
        while change > 0.0001:
            self.expectationStep()
            change=self.maximizationStep()
            print(change)
            print("Iteration {}".format(iterCounter))
            iterCounter=iterCounter+1

# Creating random data
x1, y1 = np.random.multivariate_normal(mu1, sig1, 100).T
x2, y2 = np.random.multivariate_normal(mu2, sig2, 100).T
xs = np.concatenate((x1, x2))
ys = np.concatenate((y1, y2))
data=pd.DataFrame(zip(xs,ys),columns=['X','Y'])

gmmObject=GMM(data)
gmmObject.setConfig(2)
gmmObject.iterate()

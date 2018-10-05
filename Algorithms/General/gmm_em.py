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
            'means' : [[ (max(self.data[i]) - min(self.data[i]) * y * 1.0000) / self.initialGaussianCenteres  for i in self.cols] for y in range(self.initialGaussianCenteres)],
            'deviations' : [[np.random.randint(10) for j in range(len(self.cols))] for i in range(self.initialGaussianCenteres)],
            'weights' : np.random.dirichlet(np.ones(self.initialGaussianCenteres),size=1)[0]  
        }
        print("Initial means are {}".format(self.config['means']))
        
    def prob(self,val, mu, sig, probWeight):
        p = probWeight
        #print("Val = {} and mu ={} and sig ={} and probWeight ={}".format(val,mu,sig,probWeight))
        for i in range(len(self.cols)):
            p *= scipy.stats.norm.pdf(val[i], mu[i], sig[i])
        #print("the value of p is {}".format(p))
        return p

    def getMaxLabel(self,vals):
        temp=[]
        for x in range(self.initialGaussianCenteres):
            temp.append(self.prob(vals, self.config['means'][x],self.config['deviations'][x],self.config['weights'][x]))
        return(np.argmax(temp))

    
    def expectationStep(self):
        # At this step we will compute the allocation of gaussian centres based on the current data for each point
        #self.data['label']=self.data[self.cols].apply(lambda row: np.argmax(self.prob(row.values, self.config['means'][x],self.config['deviations'][x],self.config['weights'][x]) for x in range(self.initialGaussianCenteres)) ,axis=1)        
        self.data['label']=self.data[self.cols].apply(lambda row: self.getMaxLabel(row.values),axis=1)        
        # For testing purposes
        #for y in range(100):
        #    temp=[]
        #    for x in range(self.initialGaussianCenteres):
        #        temp.append(self.prob(self.data[self.cols].loc[y].values, self.config['means'][x],self.config['deviations'][x],self.config['weights'][x]))
        #    #print("The value of temp is {}".format(temp))
        #    if(np.argmax(temp)>0):
        #        print("The value of argmax is {}".format(np.argmax(temp)))
            
        
        print("Unique Labels are {}".format(self.data['label'].unique()))
        #for curRow in range(self.data.shape[0]):
        #    probList=[self.prob(self.data[self.cols].ix[curRow].values, self.config['means'][x],self.config['deviations'][x],self.config['weights'][x]) for x in range(self.initialGaussianCenteres)]
        #    self.data['label'][curRow]=np.argmax(probList)
    
    def maximizationStep(self):
        oldMean=self.config['means']
        self.config['weights']=[
            (self.data[self.data['label']==x].shape[0]*1.0000) / self.data.shape[0] for x in range(self.initialGaussianCenteres)
        ]
        print("New means are {}".format(
        [
            [self.data[self.data['label']==x][y].values.mean() for y in self.cols] for x in range(self.initialGaussianCenteres)
        ]))
        
        self.config['means']=[
            [self.data[self.data['label']==x][y].values.mean() for y in self.cols] for x in range(self.initialGaussianCenteres)
        ]
        print("New Deviations is {}".format([
            [self.data[self.data['label']==x][y].values.std() for y in self.cols] for x in range(self.initialGaussianCenteres)
        ]))
        self.config['deviations']=[
            [self.data[self.data['label']==x][y].values.std() for y in self.cols] for x in range(self.initialGaussianCenteres)
        ]
        
        # we will calculate the change in means and return that value
        print("Means is {}".format(self.config['means']))
        print(oldMean)
        return(1)
        #return(sum([np.linalg.norm(self.config['means'][x]-oldMean[x]) for x in range(self.initialGaussianCenteres)]))
    
    def iterate(self):
        change=999
        iterCounter=0
        while change > 0.0001:
            self.expectationStep()
            change=self.maximizationStep()
            #print(change)
            #print("Iteration {}".format(iterCounter))
            iterCounter=iterCounter+1

# Creating random data
mu1=[10,10]
sig1=[[10,0],[0,10]]
mu2=[20,20]
sig2=[[10,0],[0,10]]
x1, y1 = np.random.multivariate_normal(mu1, sig1, 100).T
x2, y2 = np.random.multivariate_normal(mu2, sig2, 100).T
xs = np.concatenate((x1, x2))
ys = np.concatenate((y1, y2))
data=pd.DataFrame(zip(xs,ys),columns=['X','Y'])

gmmObject=GMM(data)
gmmObject.setConfig(2)
gmmObject.iterate()

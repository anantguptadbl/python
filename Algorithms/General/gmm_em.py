# GMM
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy

class GMM():
    def __init__(self,data):
        self.data=data
        self.cols=self.data.columns.values
        self.data['label']=9999
        self.initialGaussianCenteres=2
        
    def setConfig(self,initialGaussianCenteres):
        self.initialGaussianCenteres=initialGaussianCenteres
        self.config={
            'means' : [[ np.percentile(self.data[i],(y+1)*100/(self.initialGaussianCenteres+1)) for i in self.cols] for y in range(self.initialGaussianCenteres)],
            'deviations' : [[ self.data[j].std() for j in self.cols] for i in range(self.initialGaussianCenteres)],
            'weights' : np.random.dirichlet(np.ones(self.initialGaussianCenteres),size=1)[0]  
        }
        print("Initial means are {}".format(self.config['means']))
        print("Initial Deviations are {}".format(self.config['deviations']))
        
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
        #print("for the current values {}, the argmax is {} and prob vals are {}".format(vals,np.argmax(temp),temp))
        return(np.argmax(temp))

    
    def expectationStep(self):
        # At this step we will compute the allocation of gaussian centres based on the current data for each point
        self.data['label']=self.data[self.cols].apply(lambda row: self.getMaxLabel(row.values),axis=1)        
        #print("Unique Labels are {}".format(self.data['label'].unique()))
        
        # Resetting, as we have arrived at a non-expected convergence
        if(len(self.data['label'].unique())) < self.initialGaussianCenteres :
            print("Convergence issue, we will reset labels ")
            self.data['label'] = self.data.transform(lambda x: np.random.choice(range(self.initialGaussianCenteres), len(x)))
            self.config={
                'means' : [[self.data.loc[self.data['label']==y,i].mean()  for i in self.cols] \
                           for y in range(self.initialGaussianCenteres)],
                'deviations' : [[self.data.loc[self.data['label']==y,i].std()  for i in self.cols] \
                           for y in range(self.initialGaussianCenteres)],
                'weights' : np.random.dirichlet(np.ones(self.initialGaussianCenteres),size=1)[0]  
            }
    
    def maximizationStep(self):
        oldMean=self.config['means']
        self.config['weights']=[
            (self.data[self.data['label']==x].shape[0]*1.0000) / self.data.shape[0] for x in range(self.initialGaussianCenteres)
        ]
        self.config['means']=[
            [self.data[self.data['label']==x][y].values.mean() for y in self.cols] for x in range(self.initialGaussianCenteres)
        ]
        self.config['deviations']=[
            [self.data[self.data['label']==x][y].values.std() for y in self.cols] for x in range(self.initialGaussianCenteres)
        ]
        
        # For convergence, we are checking the diff in mean values
        print("Diff in means is {}".format(sum([np.linalg.norm(np.array(self.config['means'][x])-np.array(oldMean[x])) for x in range(self.initialGaussianCenteres)])))
        return(sum([np.linalg.norm(np.array(self.config['means'][x])-np.array(oldMean[x])) for x in range(self.initialGaussianCenteres)]))
    
    def iterate(self):
        change=999
        iterCounter=0
        while change > 0.0001:
            self.expectationStep()
            change=self.maximizationStep()
            iterCounter=iterCounter+1
            
if __name__=="__main__":
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

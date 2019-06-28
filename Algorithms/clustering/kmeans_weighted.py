### ANALYSIS 4 : Weighted Clustering
import numpy as np
import os

class WeightedKMeans(object):
    def __init__(self,data,numCentroids,valueColumns,weightColumn):
        self.data=data
        self.numCentroids=numCentroids
        self.valueColumns=valueColumns
        self.weightColumn=weightColumn
        self.weights=self.data[weightColumn]
        self.data[weightColumn]=(self.weights - min(self.weights))/(max(self.weights) - min(self.weights))
        self.data['curCentroid']=999
        
        # Max - Min : Equally placed centroid Initialization : This was causing issues
        #minValues=self.data[valueColumns].min().values
        #maxValues=self.data[valueColumns].max().values
        #self.centroids=[]
        #for y in range(self.numCentroids):
        #    curCentroid=[]
        #    for x in range(len(self.valueColumns)):
        #        curCentroid.append(minValues[x] + (y+1)*(maxValues[x] - minValues[x])/(self.numCentroids + 1))
        #    self.centroids.append(copy.deepcopy(curCentroid))
        
        # Random Initilization : This was causing issues
        #centroidInits=np.random.choice(self.data.shape[0], self.numCentroids, replace=False).tolist()
        #self.centroids=self.data.loc[np.random.choice(self.data.shape[0], self.numCentroids, replace=False).tolist()][self.valueColumns].values.tolist()
        
        # Centroid Initialization using Sharding
        # https://www.kdnuggets.com/2017/03/naive-sharding-centroid-initialization-method.html
        shardSize=int(self.data.shape[0] / self.numCentroids)
        for curCentroid in range(self.numCentroids):
            self.data.loc[range(curCentroid*shardSize,(curCentroid+1)*shardSize),'curCentroid']=curCentroid
        self.centroids=[]
        for curCentroid in range(self.numCentroids):
            self.centroids.append(np.mean(self.data[self.data['curCentroid']==curCentroid][self.valueColumns].values,axis=0))
        print("Initial Centroids are {0}".format(self.centroids))
        
    
    def compute_weighted_euclidean_distance(self,point, centroid,weight):
        return np.sqrt(np.sum((point - centroid)**2)) * weight

    def iterate_k_means(self,epochs):
        totalError=0
        
        for curEpoch in range(epochs):
            # Create a column that will measure the distance of each point from each centroid
            for centroidNum in range(self.numCentroids):
                self.data['dist{0}'.format(centroidNum)]=self.data.apply(lambda row : self.compute_weighted_euclidean_distance(row[self.valueColumns],self.centroids[centroidNum],row[self.weightColumn]),axis=1)  
            # Find out the centroid allocation for each point
            self.data['curCentroid']=self.data.apply(lambda row: np.argmin(row[['dist'+str(x) for x in range(self.numCentroids)]].values),axis=1)   
            # Find out the new centroids
            prevError=totalError
            totalError=0
            for centroidNum in range(self.numCentroids):
                # First we will create the weight columns for the points assigned to a centroid
                if(self.data[self.data['curCentroid']==centroidNum].shape[0] > 0 ):
                    weightArray=self.data.loc[self.data['curCentroid']==centroidNum,self.weightColumn].values
                    self.centroids[centroidNum]=np.matmul(weightArray,self.data[self.data['curCentroid']==centroidNum][self.valueColumns].values).reshape(-1)/sum(weightArray)
                # Calculate the error which is the sum of all the distances of all points from the assigned centroids
                if(self.data[self.data['curCentroid']==centroidNum].shape[0] >0):
                    totalError = totalError + sum(self.data[self.data['curCentroid']==centroidNum]['dist{0}'.format(centroidNum)].values)/sum(self.data[self.data['curCentroid']==centroidNum][self.weightColumn].values)             
            print("Epoch : {0} , Error : {1}".format(curEpoch,totalError))
            if(prevError >0 and abs(prevError - totalError) < 1e-5):
                print("Exiting now as priorErro : {0} curError : {1} and diff:{2}".format(prevError,totalError,prevError-totalError))
                break
        # We need to delete the centroid dist columns from the original data and just keep the centroid Number

                
data=pd.DataFrame(np.random.rand(100,3),columns=['col1','col2','abc'])
kmeansObj=WeightedKMeans(data,4,['col1','col2'],'abc')
kmeansObj.iterate_k_means(50)

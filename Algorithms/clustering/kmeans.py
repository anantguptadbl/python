### ANALYSIS 4 : Weighted Clustering
import numpy as np
import os

class WeightedKMeans(object):
    def __init__(self,data,numCentroids,valueColumns,weightColumns):
        self.data=data
        self.numCentroids=numCentroids
        self.valueColumns=valueColumns
        self.weightColumns=weightColumns
        
        # Init the Centroids
        minValues=self.data[valueColumns].min().values
        maxValues=self.data[valueColumns].max().values

        self.centroids=[]
        for y in range(self.numCentroids):
            curCentroid=[]
            for x in range(len(self.valueColumns)):
                curCentroid.append(minValues[x] + (y+1)*(maxValues[x] - minValues[x])/(self.numCentroids + 1))
            self.centroids.append(copy.deepcopy(curCentroid))
    
    
    def compute_euclidean_distance(self,point, centroid):
        return np.sqrt(np.sum((point - centroid)**2))

    def iterate_k_means(self,epochs):
        totalError=0
        self.data['curCentroid']=999
        for curEpoch in range(epochs):
            # Create a column that will measure the distance of each point from each centroid
            for centroidNum in range(self.numCentroids):
                self.data['dist{0}'.format(centroidNum)]=self.data.apply(lambda row : self.compute_euclidean_distance(row[self.valueColumns],self.centroids[centroidNum]),axis=1)  
            self.data['curCentroid']=self.data.apply(lambda row: np.argmin(row[['dist'+str(x) for x in range(self.numCentroids)]].values),axis=1)   
            for centroidNum in range(self.numCentroids):
                self.centroids[centroidNum]=np.mean(self.data[self.data['curCentroid']==centroidNum][self.valueColumns].values,axis=0)
            # Calculate the error which is the sum of all the distances of all points from the assigned centroids
            prevError=totalError
            totalError=0
            for centroidNum in range(self.numCentroids):
                totalError = totalError + sum(self.data[self.data['curCentroid']==centroidNum]['dist{0}'.format(centroidNum)].values)
            print("Epoch : {0} , Error : {1}".format(curEpoch,totalError))
            if(prevError >0 and (prevError - totalError) < 1e-5):
                print("Exiting now")
                break
                
data=pd.DataFrame(np.random.rand(100,3),columns=['col1','col2','col3'])
kmeansObj=WeightedKMeans(data,4,['col1','col2','col3'],'abc')
kmeansObj.iterate_k_means(50)

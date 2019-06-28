### ANALYSIS 4 : Weighted Clustering
import numpy as np
import os

class WeightedKMeans(object):
    def __init__(self,data,numCentroids,valueColumns,weightColumn):
        self.data=data
        self.numCentroids=numCentroids
        self.valueColumns=valueColumns
        self.weightColumn=weightColumn
        
        # Init the Centroids
        # The following logic is not working
        
        #minValues=self.data[valueColumns].min().values
        #maxValues=self.data[valueColumns].max().values
        #self.centroids=[]
        #for y in range(self.numCentroids):
        #    curCentroid=[]
        #    for x in range(len(self.valueColumns)):
        #        curCentroid.append(minValues[x] + (y+1)*(maxValues[x] - minValues[x])/(self.numCentroids + 1))
        #    self.centroids.append(copy.deepcopy(curCentroid))
        centroidInits=np.random.choice(self.data.shape[0], self.numCentroids, replace=False).tolist()
        self.centroids=self.data.loc[np.random.choice(self.data.shape[0], self.numCentroids, replace=False).tolist()][self.valueColumns].values.tolist()
    
    def compute_weighted_euclidean_distance(self,point, centroid,weight):
        return np.sqrt(np.sum((point - centroid)**2)) * weight

    def iterate_k_means(self,epochs):
        totalError=0
        self.data['curCentroid']=999
        for curEpoch in range(epochs):
            # Create a column that will measure the distance of each point from each centroid
            for centroidNum in range(self.numCentroids):
                self.data['dist{0}'.format(centroidNum)]=self.data.apply(lambda row : self.compute_weighted_euclidean_distance(row[self.valueColumns],self.centroids[centroidNum],row[self.weightColumn]),axis=1)  
            
            # Find out the centroid allocation for each point
            self.data['curCentroid']=self.data.apply(lambda row: np.argmin(row[['dist'+str(x) for x in range(self.numCentroids)]].values),axis=1)   
            print(self.data['curCentroid'].value_counts())
            # Find out the new centroids
            prevError=totalError
            totalError=0
            for centroidNum in range(self.numCentroids):
                # First we will create the weight columns for the points assigned to a centroid
                weightArray=self.data.loc[self.data['curCentroid']==centroidNum,self.weightColumn].values
                if(len(weightArray) > 0 ):
                    weightArray=(weightArray - min(weightArray)) / ( max(weightArray) - min(weightArray)).reshape(1,-1)
                    self.centroids[centroidNum]=np.matmul(weightArray,self.data[self.data['curCentroid']==centroidNum][self.valueColumns].values).reshape(-1)/sum(weights)
                    print("For epoch {0} and centroidNum as {1} the centroid is {2}".format(curEpoch,centroidNum,self.centroids[centroidNum]))
                # Calculate the error which is the sum of all the distances of all points from the assigned centroids
                if(self.data[self.data['curCentroid']==centroidNum].shape[0] >0):
                    totalError = totalError + sum(self.data[self.data['curCentroid']==centroidNum]['dist{0}'.format(centroidNum)].values)/sum(self.data[self.data['curCentroid']==centroidNum][self.weightColumn].values)
                            
            print("Epoch : {0} , Error : {1}".format(curEpoch,totalError))
            if(prevError >0 and abs(prevError - totalError) < 1e-5):
                print("Exiting now as priorErro : {0} curError : {1} and diff:{2}".format(prevError,totalError,prevError-totalError))
                break
                
data=pd.DataFrame(np.random.rand(100,3),columns=['col1','col2','abc'])
kmeansObj=WeightedKMeans(data,4,['col1','col2'],'abc')
kmeansObj.iterate_k_means(50)

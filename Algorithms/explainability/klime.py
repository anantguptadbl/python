import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import f_regression


# KLIME CLASS
class klime():
    def __init__(self,data,scoreColumn,explainColumn='explain',clusterColumn='cluster'):
        self.data=data
        self.scoreColumn=scoreColumn
        self.clusterColumn=clusterColumn
        self.explainColumn=explainColumn
        self.cols=[x for x in data.columns.values if x != self.scoreColumn and x != self.clusterColumn and x != explainColumn]
        self.metrics=[]
        self.k=1000
        
    def assignClusters(self,k):
        self.data[self.clusterColumn]=KMeans(n_clusters=k, random_state=0).fit(self.data[self.cols]).labels_

    def getLinearRegressionRSquared(self,data):
        return r2_score(
            data[self.scoreColumn].values,
            linear_model.LinearRegression().fit(data[self.cols].values,data[self.scoreColumn].values).predict(data[self.cols].values)
        )

    def getSummedRSquaredForAllClusters(self,k):
        RSquaredSum=sum([self.getLinearRegressionRSquared(self.data[self.data[self.clusterColumn]==x]) for x in range(k)])
        return([k,RSquaredSum])

    def getArrayOfKandRSquared(self):
        for k in range(10,50,10):
            self.assignClusters(k)
            self.metrics.append(self.getSummedRSquaredForAllClusters(k))

    def getLinearRegressionTop5Features(self,data):
        if(data.shape[0] > 0):
            pValues=zip(self.cols,list(f_regression(data[self.cols].values,data[self.scoreColumn].values,center=True)[1]))
            pValues=sorted(pValues,key=lambda l:l[1], reverse=True)
            #return([[x[0],x[1]] for x in pValues[0:5]])
            return(str(pValues[0:5]))
        else:
            return('') 
        
    def getFeatures(self):
        self.getArrayOfKandRSquared()
        self.k=self.metrics[np.argmax(np.array(self.metrics),axis=0)[0]][0]
        self.data[self.clusterColumn]=KMeans(n_clusters=self.k, random_state=0).fit(self.data[self.cols]).labels_
        for curCluster in range(self.k):
            variables=self.getLinearRegressionTop5Features(self.data[self.data['cluster']==curCluster])
            self.data.ix[self.data[self.clusterColumn]==curCluster,self.explainColumn]=variables
        
if __name__=="__main__":
    # Create sample data
    data=np.random.rand(10000,5)
    data=pd.DataFrame(data,columns=['col1','col2','col3','col4','score'])
    
    # Running Klime
    ko=klime(data,'score')
    ko.getFeatures()
    
    # Printing the imp features
    print(ko.data.head(5))

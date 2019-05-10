import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import f_regression

def addScores(row,columnNames):
    aggScore=0
    for curScoreColumn in columnNames:
        aggScore=aggScore + row[curScoreColumn]
    return aggScore    

# KLIME CLASS
class klimeMultiLabel():

    
    def __init__(self,data,scoreColumns,explainColumn='explain',clusterColumn='cluster'):
        self.data=data
        self.scoreColumns=scoreColumns
        self.clusterColumn=clusterColumn
        self.explainColumn=explainColumn
        self.cols=[x for x in data.columns.values if x not in self.scoreColumns and x != self.clusterColumn and x != explainColumn]
        self.metrics=[]
        self.k=1000
        # Generating the explain data for each scoreColumn
        for curScoreColumn in self.scoreColumns:
            self.data['explain' + str(curScoreColumn)]=''
        self.data['explain_local_global']=''
        self.data['explain_global_global']=''
        # Generating the aggregated score
        self.data['aggregatedScore']=self.data.apply(lambda row : addScores(row,self.scoreColumns),axis=1)
            
    
    def addScores(self,row):
        aggScore=0
        for curScoreColumn in self.scoreColumns:
            aggScore=aggScore + row[curScoreColumn]
        return aggScore    
    
    def assignClusters(self,k):
        self.data[self.clusterColumn]=KMeans(n_clusters=k, random_state=0).fit(self.data[self.cols]).labels_

    def getLinearRegressionRSquared(self,data):
        r2Score=0
        for eachScoreScolumn in self.scoreColumns:
            r2Score = r2Score + r2_score(
            data[eachScoreScolumn].values,
            linear_model.LinearRegression().fit(data[self.cols].values,data[eachScoreScolumn].values).predict(data[self.cols].values)
        )
            return r2Score

    def getSummedRSquaredForAllClusters(self,k):
        RSquaredSum=sum([self.getLinearRegressionRSquared(self.data[self.data[self.clusterColumn]==x]) for x in range(k)])
        return([k,RSquaredSum])

    def getArrayOfKandRSquared(self):
        for k in range(10,50,10):
            self.assignClusters(k)
            self.metrics.append(self.getSummedRSquaredForAllClusters(k))

    def getLinearRegressionTop5Features(self,data,scoreColumn):
        if(data.shape[0] > 0):
            pValues=zip(self.cols,list(f_regression(data[self.cols].values,data[scoreColumn].values,center=True)[1]))
            pValues=sorted(pValues,key=lambda l:l[1], reverse=True)
            return(str(pValues[0:5]))
        else:
            return('') 
        
    def getFeatures(self):
        self.getArrayOfKandRSquared()
        self.k=self.metrics[np.argmax(np.array(self.metrics),axis=0)[0]][0]
        self.data[self.clusterColumn]=KMeans(n_clusters=self.k, random_state=0).fit(self.data[self.cols]).labels_
        for curCluster in range(self.k):
            for curScoreColumn in self.scoreColumns:
                variables=self.getLinearRegressionTop5Features(self.data[self.data['cluster']==curCluster],curScoreColumn)
                self.data.ix[self.data[self.clusterColumn]==curCluster,'explain' + str(curScoreColumn)]=variables
        for curCluster in range(self.k):
            variables=self.getLinearRegressionTop5Features(self.data[self.data['cluster']==curCluster],'aggregatedScore')
            self.data.ix[self.data[self.clusterColumn]==curCluster,'explain_local_global']=variables
        variables=self.getLinearRegressionTop5Features(self.data,'aggregatedScore')
        self.data['explain_global_global']=variables
        print("After adding all the scores")
        print(self.data.head(5))
        
if __name__=="__main__":
    # Create sample data
    data=np.random.rand(10000,6)
    data=pd.DataFrame(data,columns=['col1','col2','col3','col4','score1','score2'])
    
    # Running Klime
    ko=klimeMultiLabel(data,['score1','score2'])
    print("Generated the KLime Object")
    
    ko.getFeatures()
    
    # Printing the imp features
    print(ko.data.head(5))

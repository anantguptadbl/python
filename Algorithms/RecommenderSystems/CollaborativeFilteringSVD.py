from scipy.sparse import linalg as slinalg
from scipy import sparse
import copy
import pandas as pd
import scipy
import numpy as np
from sklearn.cluster import KMeans
from scipy.stats import rankdata

class CollaborativeFiltering():
    def __init__(self,trainData,testData,indexCol,columnCol,valueCol,categoryData,categoryColumn):
        self.trainData=trainData
        self.testData=testData
        self.indexCol=indexCol
        self.columnCol=columnCol
        self.valueCol=valueCol
        self.categoryData=categoryData
        self.categoryCol=categoryColumn
        
    def getUserEmbeddings(self):
        self.userEmbeddings=np.dot(self.U,self.S)
        
    def getProductEmbeddings(self):
        self.productEmbeddings=np.dot(self.S,self.V).T
        
    def convertToSVDData(self):
        self.inpArrayDense=self.trainData.pivot(index=self.indexCol,columns=self.columnCol,values=self.valueCol).fillna(0)
        self.userVals=self.inpArrayDense.index.values
        self.productVals=self.inpArrayDense.columns.values
        self.inpArray=sparse.csr_matrix(self.inpArrayDense).asfptype()
        
    def getMinSingularVectorsMethodValMag(self,threshold=0.8):
        U,S,V=scipy.sparse.linalg.svds(self.inpArray,k=min(self.inpArray.shape[0]-1,self.inpArray.shape[1]-1))
        S1=list(reversed(S))
        cumSum=np.cumsum(S1) / sum(S1)
        minVal=min([i for i,x in enumerate(cumSum) if x >= threshold])
        self.SingularValuesNumber=len(S1[0:minVal])
        
    def getUSV(self,k):
        inpArray=sparse.csr_matrix(self.inpArray).asfptype()
        U,S,V=scipy.sparse.linalg.svds(self.inpArray,k=k)
        self.SArray=copy.deepcopy(S)
        S=np.diag(S)
        self.U=U
        self.V=V
        self.S=S
        
    def getUserClusters(self,k=10):
        self.userClusters=pd.DataFrame(zip(KMeans(n_clusters=k,random_state=0).fit(self.userEmbeddings).labels_,self.userVals,self.userEmbeddings),
                                       columns=['userCluster','user','userEmbedding'])
                                       
    def getProductClusters(self,k=10):
        self.productClusters=pd.DataFrame(zip(KMeans(n_clusters=k,random_state=0).fit(self.productEmbeddings).labels_,self.productVals,self.productEmbeddings),
                                       columns=['productCluster','product','productEmbedding'])
                
    def getUserSuggestions(self,curUser,userQuantileCutOff=0.4):
        curUserCluster=self.userClusters[self.userClusters['user']==curUser]['userCluster'].values[0]
        curUserEmbedding=self.userClusters[self.userClusters['user']==curUser]['userEmbedding'].values[0]
        curUserDist=self.userClusters[self.userClusters['userCluster']==curUserCluster][['user','userEmbedding']]
        curUserDist['dist']=curUserDist['userEmbedding'].map(lambda x : scipy.spatial.distance.euclidean(x,curUserEmbedding))
        del(curUserDist['userEmbedding'])
        if(curUserDist.shape[0] >0):
            curUserDist['minMaxDist']=(curUserDist['dist'].values - (min(curUserDist['dist'].values * 1.00))) / (max(curUserDist['dist'].values) - min(curUserDist['dist'].values))
            curUserDist['RankDist']=rankdata(curUserDist['dist'].values,method='min')
            DistCutOff10PctQuantileUser=curUserDist.quantile(userQuantileCutOff)[0]
            return(curUserDist[curUserDist['dist'] <= DistCutOff10PctQuantileUser][['user','dist','minMaxDist','RankDist']].values)
        else:
            return([])
        
    def getProductSuggestions(self,curProduct,productQuantileCutOff=0.4):
        curProductCluster=self.productClusters[self.productClusters['product']==curProduct]['productCluster'].values[0]
        curProductEmbedding=self.productClusters[self.productClusters['product']==curProduct]['productEmbedding'].values[0]
        curProductDist=self.productClusters[self.productClusters['productCluster']==curProductCluster][['product','productEmbedding']]
        curProductDist['dist']=curProductDist['productEmbedding'].map(lambda x : scipy.spatial.distance.euclidean(x,curProductEmbedding))
        del(curProductDist['productEmbedding'])
        if(curProductDist.shape[0] >0):
            curProductDist['minMaxDist']=(curProductDist['dist'].values - (min(curProductDist['dist'].values * 1.00))) / (max(curProductDist['dist'].values) - min(curProductDist['dist'].values))
            curProductDist['RankDist']=rankdata(curProductDist['dist'].values,method='min')
            DistCutOff10PctQuantileProduct=curProductDist.quantile(productQuantileCutOff)[0]
            return(curProductDist[curProductDist['dist'] <= DistCutOff10PctQuantileProduct][['product','dist','minMaxDist','RankDist']].values)
        else:
            return([])
        
    def getReferenceUserProducts(self,curUser):
        return(self.trainData[self.trainData[self.indexCol]==curUser][[self.indexCol,self.columnCol,self.valueCol]].values)
    
    def getAllSuggestions(self,curUser):
        suggestedProducts=[]
        for userSuggestion in self.getUserSuggestions(curUser):
            for curProduct in self.getReferenceUserProducts(userSuggestion[0]):
                for productSuggestion in self.getProductSuggestions(curProduct[1]):
                    suggestedProducts.append([
                        userSuggestion[0],userSuggestion[1],userSuggestion[2],userSuggestion[3],
                        curProduct[1],curProduct[2],
                        productSuggestion[0],productSuggestion[1],productSuggestion[2],productSuggestion[3]
                                             ])
        return(suggestedProducts)
    
    def compare2Arrays(self,arr1,arr2):
        count=0
        for x in range(len(arr1)):
            if((arr1[x] >=1 and (abs(arr1[x] - arr2[x]) < ((arr1[x]*1.000)/2))) or (arr1[x]==0 and abs(arr2[x]) < 0.3) ):
                count=count+1
        if(count > (len(arr1) * 0.8)):
            return True
        else:
            return False
        
    def getUserProductionRecreationMatches(self,recreatedData):
        userComparison=[]
        productComparison=[]
        for curRow in range(self.inpArrayDense.shape[0]):
            userComparison.append(self.compare2Arrays(self.inpArrayDense.values[curRow],recreatedData[curRow]))
        for curCol in range(self.inpArrayDense.shape[1]):
            productComparison.append(self.compare2Arrays(self.inpArrayDense.values[:,curCol],recreatedData[:,curCol]))
        return([userComparison,productComparison])
    
    def getComparisonStatsbyIncSV(self,incrementCounter=10):
        fullComparison=[]
        for singVectNumbers in range(1,len(self.SArray)+1,incrementCounter):
            if(singVectNumbers==1):
                S1=np.diag([self.SArray[0]] + [0])
            else:
                S1=np.diag(self.SArray[0:singVectNumbers-1])
            for x in range(self.SingularValuesNumber - S1.shape[0]):
                S1.np.vstack((S1,np.zeros(S1.shape[1])))
            for x in range(self.SingularValuesNumber -S1.shape[1]):
                S1=np.hstack((S1,np.zeros(S1.shape[0]).T.reshape(S1.shape[0],1)))
            recreatedData=np.dot(np.dot(self.U,S1),self.V)
            fullComparison.append(self.getUserProductRecreationMatches(recreatedData) + [singVectNumbers])
        return(fullComparison)
    
    def getRecommendations_Algo1(self,userList):
        results=dict((user,self.getAllSuggestions(user)) for user in userList)
        fullData=pd.DataFrame(columns=['refUser','refUserDist','refUserminMaxDist','refUserRankDist','refUserProduct','refUserProductEventCount','refProduct','refProductDist','refProductminMaxDist','refProductRankDist','user']) 
        for curUser in results:
            curData=pd.DataFrame(results[curUser],columns=['refUser','refUserDist','refUserminMaxDist','refUserRankDist','refUserProduct','refUserProductEventCount','refProduct','refProductDist','refProductminMaxDist','refProductRankDist'])
            curData['user']=curUser
            fullData=pd.concat([fullData,curData])
        self.getCategoryRecommendations_Algo1(fullData)
        fullData['reason_code']=fullData.apply(lambda row : self.getRowRecommendation(row),axis=1)
        return([fullData,self.categoryScore])
    
    def getCategoryRecommendations_Algo1(self,recommendData):
        r1=recommendData.merge(self.categoryData,left_on='refProduct',right_on=self.columnCol,how='left')
        self.categoryScore=r1.groupby(['user',self.categoryCol],as_index=False)['refProductminMaxDist'].mean().sort_values('refProductminMaxDist',ascending=False)
        self.categoryScore.columns=['user',self.categoryCol,'category_score']

    
    def getRowRecommendation(self,row):
        if(row['user']==row['refUser']):
            if(row['refProduct']==row['refUserProduct']):
                return("User has acted on {} in the past".format(row['refProduct']))
            else:
                return("The product {} is similar to the product {} which user has acted on in the past".format(row['refProduct'],row['refUserProduct']))
        else:
            if(row['refProduct']==row['refUserProduct']):
                return("Similar user {} have acted on {} in the past".format(row['refUser'],row['refProduct']))
            else:
                return("the {} is similar to {} which a similar user {} have acted on in the past".format(row['refProduct'],row['refUserProduct'],row['refUser']))
      
    def compareRecommendations(self,fullData):
        testResults=self.testData.merge(fullData,left_on=[self.indexCol,self.columnCol],right_on=['user','refProduct'],how='left')
        testResults=testResults.merge(self.categoryData,left_on=self.columnCol,right_on=self.columnCol,how='left')
        testResults=testResults.merge(self.categoryScore,left_on=['user',self.categoryCol],right_on=['user',self.categoryCol],how='left')
        testResults=testResults[[self.indexCol,self.columnCol,'refProduct',self.categoryCol,'category_score']].drop_duplicates()
        # Get Metrics at the Product Level
        testResultsStats=pd.DataFrame(zip(testResults.groupby(self.indexCol)[self.columnCol].count().index.values,testResults.groupby(self.indexCol)[self.columnCol].count().values,testResults.groupby(self.indexCol)['refProduct'].count().values),columns=[self.indexCol,'ideaEventCountOriginal','ideaEventCountPredicted'])
        if(testResultsStats.shape[0] >0):
            testResultsStats['percCovered']=testResultsStats.apply(lambda x: 0 if x['ideaEventCountPredicted']==0 else  (x['ideaEventCountPredicted'] * 1.000) / x['ideaEventCountOriginal'],axis=1)
            AvgMatchLessEvents=testResultsStats[testResultsStats['ideaEventCountOriginal'] <=2]['percCovered'].mean()
            AvgMatchMoreEvents=testResultsStats[testResultsStats['ideaEventCountOriginal']  > 2]['percCovered'].mean()
        else:
            AvgMatchLessEvent=0
            AvgMatchMoreEvents=0
        # Get Metrics at the Category Level
        testResultsCategoryStats=pd.DataFrame(zip(testResults.groupby(self.indexCol)[self.columnCol].count().index.values,testResults.groupby(self.indexCol)[self.categoryCol].count().values,testResults.groupby(self.indexCol)['category_score'].count().values),columns=[self.indexCol,'categoryEventCountOriginal','categoryPredicted'])
        if(testResultsCategoryStats.shape[0] >0):
            testResultsCategoryStats['percCovered']=testResultsCategoryStats.apply(lambda row: 0 if row['categoryEventCountOriginal']==0 else (row['categoryPredicted']* 1.000) / row['categoryEventCountOriginal'],axis=1)
            AvgMatchLessEventsCategory=testResultsCategoryStats[testResultsCategoryStats['categoryEventCountOriginal'] <=2]['percCovered'].mean()
            AvgMatchMoreEventsCategory=testResultsCategoryStats[testResultsCategoryStats['categoryEventCountOriginal']  > 2]['percCovered'].mean()
        else:
            AvgMatchLessEventsCategory=0
            AvgMatchMoreEventsCategory=0
        return([testResults,testResultsStats,AvgMatchLessEvents,AvgMatchMoreEvents,AvgMatchLessEventsCategory,AvgMatchMoreEventsCategory])
    
    
def runCollab(categoryData,categoryColumn,inputData,inputDataTest,userList,indexCol,columnCol,valCol):
    print("entered run collab")
    cfObject=CollaborativeFiltering(inputData,inputDataTest,indexCol,columnCol,valCol,categoryData,categoryColumn)
    cfObject.convertToSVDData()
    cfObject.getMinSingularVectorsMethodValMag()
    #cfObject.SingularValuesNumber=50
    cfObject.getUSV(cfObject.SingularValuesNumber)
    cfObject.getUserEmbeddings()
    cfObject.getProductEmbeddings()
    cfObject.getUserClusters(2)
    cfObject.getProductClusters(2)
    [recommendData,recommendCategory]=cfObject.getRecommendations_Algo1(userList)
    recommendTestResults=cfObject.compareRecommendations(recommendData)
    print("The recommendations power for users with count <=2 is {} and >2 is {}".format(recommendTestResults[2],recommendTestResults[3]))
    print("The category recommendation power for users with category count <=1 is {} and with >1 is {}".format(recommendTestResults[4],recommendTestResults[5]))
    return([recommendData,recommendCategory,recommendTestResults[0],recommendTestResults[2],recommendTestResults[3],recommendTestResults[4],recommendTestResults[5]])


sampleData=pd.DataFrame([['user1','prod1',1],['user1','prod2',2],['user2','prod1',1],['user3','prod4',1],['user4','prod1',1],['user4','prod3',1],['user5','prod1',1],['user5','prod2',1]],columns=['user','product','count'])
sampleDataTest=pd.DataFrame([['user1','prod1',1],['user2','prod3',1]],columns=['user','product','count'])
sampleProductCategory=pd.DataFrame([['prod1','1'],['prod2','1'],['prod3',2],['prod4',2]],columns=['product','category'])
runCollab(
    sampleProductCategory,
    'category',
    sampleData,
    sampleDataTest,
    ['user1'],
    'user',
    'product',
    'count'
)

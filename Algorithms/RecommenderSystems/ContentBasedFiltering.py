# Content based filtering for Recommender systems


class ContentBasedFiltering():
    def __init__(self,isContentFeaturesBinary,contentFeatures,contentCol,contentFeaturesColumns,interactions,interactionUserCol,interactionItemCol,interactionValueCol):
        #self.contentFeatureColumns=['f1','f2','f3','f4']
        self.contentFeatureColumns=contentFeaturesColumns
        self.contentCol=contentCol
        self.contentFeatures=contentFeatures[[contentCol] + self.contentFeatureColumns]
        self.dataArr=interactions
        self.items=self.contentFeatures[contentCol].values
        self.isContentFeaturesBinary=isContentFeaturesBinary
        self.interactionUserCol=interactionUserCol
        self.interactionItemCol=interactionItemCol
        self.interactionValueCol=interactionValueCol

    def rankPct(self,df,colList):
        for curCol in colList:
            df[curCol]=df[curCol].rank(pct=True)
        return(df)
    
    def getSuggestions(self):
        # We will do it only when item features are non-binary
        if(self.isContentFeaturesBinary==False):
            self.contentFeatures=self.rankPct(self.contentFeatures,self.contentFeatureColumns)
        self.contentFeatures[self.contentCol]=self.items
        self.contentFeatures['aggMetric']=self.contentFeatures.apply(lambda row : sum([row[x] for x in self.contentFeatureColumns]),axis=1)
        for curCol in self.contentFeatureColumns:
            self.contentFeatures[curCol]=self.contentFeatures.apply(lambda row : 0 if row['aggMetric']==0 else ((1.000 * row[curCol]) / np.sqrt(row['aggMetric'])),axis=1)
        
        # We will do this only when item features are binary
        if(self.isContentFeaturesBinary==True):
            IDFValues=self.getIDFValues()
        tempResults=self.dataArr.merge(self.contentFeatures,left_on=self.interactionItemCol,right_on=self.interactionItemCol,how='inner')
        for curCol in self.contentFeatureColumns:
            tempResults[curCol]=tempResults.apply(lambda row : row[self.interactionValueCol] * row[curCol],axis=1)
        
        self.userProfile=tempResults.groupby(self.interactionUserCol)[self.contentFeatureColumns].sum()
        if(self.isContentFeaturesBinary==True):
            for i,curCol in enumerate(self.contentFeatureColumns):
                self.userProfile[curCol]=[x * IDFValues[i] for x in self.userProfile[curCol]]
        
        contentPredictions=np.dot(self.contentFeatures[self.contentFeatureColumns].values,self.userProfile.T)
        contentPredictions=pd.DataFrame(contentPredictions,columns=self.userProfile.index.values)
        contentPredictions.index=self.contentFeatures[self.interactionItemCol].values
        contentPredictions=contentPredictions.T
        contentPredict=[]
        for i,row in enumerate(contentPredictions):
            for j,column in enumerate(contentPredictions[i]):
                contentPredict.append([i,j,contentPredictions[i][j]])
        self.contentPredict=pd.DataFrame(contentPredict,columns=[self.interactionItemCol,self.interactionUserCol,'contentScore'])
        self.contentPredict.pivot(index=self.interactionUserCol,columns=self.interactionItemCol,values='contentScore')
        return(self.contentPredict)

    def getIDFValues(self):
        IDFValues=[]
        for curCol in self.contentFeatureColumns:
            IDFValues.append(np.log(1.0000 * self.contentFeatures.shape[0]) / self.contentFeatures[self.contentFeatures[curCol] !=0].shape[0])
        return(np.array(IDFValues))

if __name__=="__main__":
    contentFeatures=pd.DataFrame([
        [0,1,1,1,1],
        [1,0,1,0,1],
        [2,0,0,1,1],
        [3,0,0,1,1],
        [4,0,1,0,1],
        [5,0,1,1,1]
    ],columns=['item','f1','f2','f3','f4'])

    arr=np.array([[1,0,1,0,0,0],
                 [0,1,0,1,0,0],
                 [1,1,1,1,0,0],
                 [1,0,1,0,0,0],
                 [0,0,0,0,1,1],
                 [1,0,1,0,1,1],
                 [1,0,1,0,0,0],
                 [1,0,1,0,1,1]])
    dataArr=[]
    for i,row in enumerate(arr):
        for j,col in enumerate(arr[i]):
            dataArr.append([i,j,arr[i][j]])
    
    dataArr=pd.DataFrame(dataArr,columns=['user','item','score'])
    CBF=ContentBasedFiltering(True,contentFeatures,'item',['f1','f2','f3','f4'],dataArr,'user','item','score')
    suggestions=CBF.getSuggestions()
    print(suggestions.pivot(index='user',columns='item',values='contentScore'))

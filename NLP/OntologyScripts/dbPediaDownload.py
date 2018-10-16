# import os
import json
import itertools
from SPARQLWrapper import SPARQLWrapper, JSON
import numpy as np
import json
import pandas as pd
import threading


class getDataFromDBPedia():
    def __init__(self,typeString,workFolder):
        self.typeString=typeString
        self.workFolder=workFolder
        self.status={}
        self.invidualItemStatus='incomplete'
        self.invidualItemLastLetter='AA'
        self.invidualItemLastFile=0
        
        # We will write a dummy JSON status file if not already present, so that it need not be ahndled every time
        if os.path.exists(workFolder + '/status.json'):
            print("We already have a prior status")
        else:
            self.status['type']=self.typeString
            self.status['individualItemStatus']={}
            self.status['individualItemStatus']['status']=self.invidualItemStatus
            self.status['individualItemStatus']['lastLetter']=self.invidualItemLastLetter
            self.status['individualItemStatus']['lastFile']=self.invidualItemLastFile
            with open(workFolder + '/status.json','w') as f:
                f.write(json.dumps(self.status))
        
    def getStatusInividualItems(self):
        # Check whether we already have downloaded some data in the existing workfolder
        with open(self.workFolder + '/status.json') as f:
            self.status = json.load(f)
        if( 'type' in self.status):
            if(self.status['type'] != self.typeString):
                return False
        if( 'individualItemStatus' in self.status):
            # This means that we have prior downloaded
            self.invidualItemStatus=self.status['individualItemStatus']['status']
            self.invidualItemLastLetter=self.status['individualItemStatus']['lastLetter']
            self.invidualItemLastFile=self.status['individualItemStatus']['lastFile']
            
        return True
    
    def getIndividualItems(self):
        firstCharacterString='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@_#$^&='
        characterList=list(itertools.product(firstCharacterString,firstCharacterString))
        characterList=[str(x[0]) + str(x[1]) for x in characterList]
        fullCompanyData=pd.DataFrame()
        
        for firstThreeCharacters in characterList[characterList.index(self.invidualItemLastLetter):]:
            query="""SELECT distinct ?orgObject ?orgLabel
            WHERE {
            ?orgObject rdf:type placeholderObjectType .
            ?orgObject rdfs:label ?orgLabel .
            filter langMatches(lang(?orgLabel),"en")
            filter(regex(REPLACE(?orgLabel, '"', '', "i"),"^%s","i"))
            }
            """ % format(firstThreeCharacters)
            query=query.replace('placeholderObjectType',self.typeString)
            data=self.returnDF(self.get_dbpedia_sparql_data(query))
            print("{} : {}".format(firstThreeCharacters,data.shape[0]))
            fullCompanyData=pd.concat([fullCompanyData,data])
            if(fullCompanyData.shape[0] > 50000):
                print("Writing data to file {}".format(self.invidualItemLastFile))
                fullCompanyData.to_csv(self.workFolder + '/FullData{}.csv'.format(self.invidualItemLastFile),index=False,encoding='utf-8')
                self.invidualItemLastFile=self.invidualItemLastFile + 1
                fullCompanyData=pd.DataFrame()
                
                # We will refresh the status JSON and write it back
                self.invidualItemStatus='incomplete'
                self.invidualItemLastLetter=firstThreeCharacters
                self.status['individualItemStatus']['status']=self.invidualItemStatus
                self.status['individualItemStatus']['lastLetter']=self.invidualItemLastLetter
                self.status['individualItemStatus']['lastFile']=self.invidualItemLastFile
                with open(self.workFolder + '/status.json','w') as f:
                    f.write(json.dumps(self.status))
        
        print("Writing data to file {}".format(self.fileCounter))
        fullCompanyData.to_csv(self.workFolder + '/FullData{}.csv'.format(self.invidualItemLastFile),index=False,encoding='utf-8')
        self.invidualItemStatus='complete'
        self.invidualItemLastLetter=firstThreeCharacters
        self.invidualItemLastFile=fileCounter
        self.status['individualItemStatus']['status']=self.invidualItemStatus
        self.status['individualItemStatus']['lastLetter']=self.invidualItemLastLetter
        self.status['individualItemStatus']['lastFile']=self.invidualItemLastFile
        with open(self.workFolder + '/status.json','w') as f:
            f.write(json.dumps(self.status))

    def get_dbpedia_sparql_data(self,query):
        sparql = SPARQLWrapper("http://dbpedia.org/sparql")
        sparql.setReturnFormat(JSON)
        sparql.setQuery(query)  # the previous query as a literal string
        return sparql.query().convert()

    def returnDF(self,data):
        dataArray=[]
        colNames=data['head']['vars']
        for row in data['results']['bindings']:
            rowArray=[]
            for colName in colNames:
                rowArray.append(row[colName]['value'])
            dataArray.append(rowArray)

        dataArray=pd.DataFrame(dataArray,columns=colNames)
        return(dataArray)
            
if __name__=="__main__":
    ThingObject=getDataFromDBPedia('owl:Thing','/home/anantgupta/Documents/Programming/dbpedia/Thing')
    if(ThingObject.getStatusInividualItems()==True):
        ThingObject.getIndividualItems()
    
# After we have got the ORG LIST, we will now be exporting the entire ORG Data Structure available in DBPEDIA
import os
fullData=pd.DataFrame()
for x in os.listdir('/home/anantgupta/Documents/Programming/dbpedia/Thing'):
    fullData=pd.concat([fullData,pd.read_csv('/home/anantgupta/Documents/Programming/dbpedia/Thing/'+str(x))])
    print(fullData.shape)

fullData.to_csv('/home/anantgupta/Documents/Programming/dbpedia/Thing/ThingFullList.txt',index=False)

# Get the details around each Thing
orgFullData=pd.read_csv('/home/anantgupta/Documents/Programming/dbpedia/Thing/ThingFullList.txt')

orgDetailedData=pd.DataFrame(columns=['org','property','value'])
batch='A'
batchCounter=0
for iterCounter,x in enumerate(orgFullData['orgObject'].values):
    query="""
    SELECT ?property ?hasValue
    WHERE {
      <%s> ?property ?hasValue
    }
    """ % (x)
    tempData=returnDF(get_dbpedia_sparql_data(query))
    orgDetailedData=pd.concat([orgDetailedData,pd.DataFrame(zip([x] * tempData.shape[0],tempData['property'].values,tempData['hasValue'].values),columns=['org','property','value'])])    
    if(iterCounter % 1000 ==0):
        orgDetailedData.to_csv('/home/anantgupta/Documents/Programming/dbpedia/Thing/' + batch+'_'+str(batchCounter)+'.csv',index=False,encoding='utf-8')
        print("Finished counter {} and batch {}".format(batchCounter,batch))
        orgDetailedData=pd.DataFrame(columns=['org','property','value'])
        batchCounter=batchCounter+1

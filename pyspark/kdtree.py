# KDTREE CLASS

def getMedian(inputDF,colName):
    # Currently the third argument will be set to 0 since, we want the actual values
    # The 0.5 denotes the median value
    #from pyspark.sql.functions import udf
    #squared_udf = udf(squared, LongType())
    #df = sqlContext.table("test")
    #display(df.select("id", squared_udf("id").alias("id_squared")))
    #medianUDF=udf(getMedian,IntegerType())
    return inputDF.approxQuantile(colName, [0.5], 0)[0]

class dNode():
    def __init__(self):
        self.conditions=[]
        self.leafNode=False
        
    def addLeftFilter(self,condition):
        self.left.conditions.append(condition)
        
    def addRightFilter(self,condition):
        self.right.conditions.append(condition)

    def createLeftRightNodes(self):
        self.left=dNode()
        self.right=dNode()
        self.left.conditions.extend(self.conditions)
        self.right.conditions.extend(self.conditions)
    
    def markAsLeafNode(self):
        self.leafNode=True
        
    def isLeaf(self):
        return self.leafNode
    
# Recursive Function to populate the condition tree
def recursiveCall(counter,dNodeObject,sparkDF,colList):
    if(counter == len(colList)):
        dNodeObject.markAsLeafNode()
        return
    else:
        curCol=colList[counter]
        curMedianVal=getMedian(sparkDF,curCol)
        dNodeObject.createLeftRightNodes()
        dNodeObject.addLeftFilter(curCol + ' < ' + str(curMedianVal))
        dNodeObject.addRightFilter(curCol + ' >= ' + str(curMedianVal))
        recursiveCall(counter+1,dNodeObject.left,sparkDF.filter(' and '.join(dNodeObject.left.conditions)),colList)
        recursiveCall(counter+1,dNodeObject.right,sparkDF.filter(' and '.join(dNodeObject.right.conditions)),colList)

# Function to get the Recursive Leaf Conditions
def getRecursiveLeafConditions(dNodeObject):
    if(dNodeObject.isLeaf()==True):
        leafConditions.append(dNodeObject.conditions)
    else:
        getRecursiveLeafConditions(dNodeObject.left)
        getRecursiveLeafConditions(dNodeObject.right)

# Execute the recursive call
origObject=dNode()
colList=['Day_of_Month','Day_of_Week','week_of_month','mean_Hour_of_day']
recursiveCall(0,origObject,df,colList)

# Final Leaf Node Conditions
leafConditions=[]
getRecursiveLeafConditions(origObject)
leafConditions=[[i,' and '.join(x)] for i,x in enumerate(leafConditions)]

# We will register the table as tempTable
df.registerTempTable("TempTableDF")

# Recreate the KDTree Group Query
fullQuery='select {},case '.format(','.join(df.columns))
for x in leafConditions:
    fullQuery=fullQuery + ' when {} then {} '.format(x[1],x[0])
fullQuery=fullQuery + 'else 99999 end as kdtreeGroup from TempTableDF'

# Assign it back to the original DF object
df=sqlContext.sql(fullQuery)

# RePartition the data at kdtreeGroup level
df=df.repartition(max([x[0] for x in leafConditions]) + 1,df.kdtreeGroup)

# Sample Data
df.take(5)

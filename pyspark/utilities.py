
def dropFirstRowPartitionFunction(index,iterator):
    # Solution 1
    #return iter(list(iterator)[1:]) if index==0 else iterator
    # Solution 2
    if(index==0):
        for subIndex,item in enumerate(iterator):
            if subIndex > 0:
                yield item
    else:
        yield iterator

# Function to drop first row
def dropfirstRow(rddObject):
    return(rddObject.mapPartitionsWithIndex(dropFirstRowPartitionFunction))

# Function to create a row index
def generateRowIndex(dfObject,indexColName):
    dfObject=dfObject.withColumn(indexColName,lit(1))
    curSelect=F.row_number().over(Window.partitionBy(indexColName).orderBy(indexColName))
    dfObject=dfObject.withColumn(indexColName,curSelect)
    return(dfObject)
    
if __name__=="__main__":
    data=sc.TextFile("file:///someFile.txt")
    data=dropfirstRow(data)


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

def dropfirstRow(rddObject):
    return(rddObject.mapPartitionsWithIndex(dropFirstRowPartitionFunction))
  
if __name__=="__main__":
    data=sc.TextFile("file:///someFile.txt")
    data=dropfirstRow(data)

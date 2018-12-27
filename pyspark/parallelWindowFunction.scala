def getRankedData(rankMap:Map[String,Array[String]],partitionString:String,inputDataFrame:DataFrame,inputDataFrameTempTable:String,tempTableAlias:String):DataFrame =
{

// Partition by on the partition String
inputDataFrame.repartition(col(partitionString))

//Create the DataFrames
val rankMapPar = rankMap.par
rankMappar.tasksupport=new ForkJoinTaskSupport(new scala.concurrent.forkjoin.ForkJoinPool(10))
rankMapPar.keys.foreach
{
  colName =>
    val curColName=rankMapPar.get(colName).get(0)
    val AscDesc=rankMapPar.get(colName).get(1)
    if(AscDesc=="desc")
    {
      val curRankDF=inputDataFrame.select(col(partitionString),col(curColName)).withColumn(colName,rank().over(Window.partitionBy(col(partitionString)).orderBy(col(curColName).desc)))
      curRankDF.registerTempTable(tempTableAlias + colName)
      curRankDF.cache()
    }
    else
    {
      val curRankDF=inputDataFrame.select(col(partitionString),col(curColName)).withColumn(colName,rank().over(Window.partitionBy(col(partitionString)).orderBy(col(curColName).asc)))
      curRankDF.registerTempTable(tempTableAlias + colName)
      curRankDF.cache()
    }
}

var joinString=" from " + inputDataFrameTempTable + " "
rankMap.keys.foreach
{
  colName => columnString = columnString + "," + tempTableAlias + colName + "." + colName
}

// Execute the joining
println("The final query is " + columnString + joinString)
val rankData=SparkSetup.hive.sql(columnString + joinString)
rankData
}

# Emthod using Futures to write a dataframe to parquet location 
def startWriting(dataFrameObject:DataFrame, fileName : String)
{
  val f1=Future
  {
    dataFrameObject.write.mode(SaveMode.Overwrite).parquet(location + fileName)
  }
  f1.onComplete
  {
    case Sucess(value) =>
    {
      println("The dataframe has been successfully written to parquet file")
    }
    case Failure(e) => 
    {
      println("The dataframe write has been unsuccessful with the following error message" + e.toString())
    }
  }
}

import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.StringIndexerModel
import scala.collection.mutable.ArrayBuffer
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.types.StringType

data.take(5).foreach(println)
data.printSchema()
val stageArray=new ArrayBuffer[StringIndexerModel]
val colNames=data.columns.toSeq
colNames.foreach
{
  x=>
  if(data.schema(x).dataType == StringType)
  {
    stageArray += new StringIndexer().setInputCol(x).setOutputCol(x + "_index").fit(data)
  }
} // We have created a list of pipeline models for string indexer

val pipelineModel=new Pipeline().setStages(stageArray.toArray)
val data2=pipelineModel.fit(data).transform(data)
data2.take(5).foreach(println)
data2.printSchema()

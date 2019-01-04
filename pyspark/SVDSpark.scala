// We will assume that data is a dataframe with 3 columns that we want to leverage for SVD
// 1) rowVal
// 2) columnVal
// 3) metricVal

import scala.math.min

data.cache()
data.printSchema()

val stageArray=newArrayBuffer[StringIndexerModel]
val colNames=data.columns.toSeq
colNames.foreach
{
x=>
if(data.schema(x).dataType == StringType)
  {
    stageArray += new StringIndexer().setInputCol(x).setOutputCol(x +"_index").setHandleInvalid("skip").fit(data)
  }
}

val pipelineModel=new Pipeline().setStages(stageArray.toArray)
data2=pipelineModel.fit(data).transform(data)

val datax=data2.groupBy("rowVal_index").pivot("columnVal_index").agg(sum("secNav")).na.fill(0.0).drop($"rowVal_index")
val curSchema=datax.schema

// This is to remove the first row
val rdd = datax.rdd.mapPartitionsWithIndex
{
  case (index,iterator) => if(index==0) iterator.drop(1) else iterator
}

val datay= sqlContext.createDataFrame(rdd,curSchema)
val data4=datay.rdd.map
{
  row =>
  val array=row.toSeq.toArray
  Vectors.dense(array.map(_.asInstanceOf[BigDecimal].doubleValue()))
}


val mat:RowMatrix = new RowMatrix(data4)
val numberOfVectors=min(mat.numCols(),mat.numRows())
val svd:SingularValueDecomposition[RowMatrix,Matrix]=mat.computeSVD(numberOfVectors.toInt,computeU=true)
val U:RowMatrix = svd.U
val S:Vector = svd.s
val V:Matrix = svd.V

// We now have the USV matrices from the SVD

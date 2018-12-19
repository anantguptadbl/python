val accum=sc.accumulator(0,"myAccum")

someDF.foreach
{
x=>
if(scala.math.BigDecimal(x.getAs[java.math.BigDecimal]("featureVal")) > 0) accum+=1
}

println("The final value of accum is"  + accum)

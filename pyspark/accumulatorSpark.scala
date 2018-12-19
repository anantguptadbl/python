val accum=sc.accumulator(0,"myAccum")

someDF.foreach
{
x=>
if(x.getAs[scala.math.BigDecimal]("featureVal") > 0) accum+=1
}

println("The final value of accum is"  + accum)

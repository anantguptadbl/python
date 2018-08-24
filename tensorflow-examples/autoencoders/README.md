# AutoEncoders

AutoEncoders are one of the simplest and powerful tools in neural networks. I have tried to provide examples of various kinds of autoencoders

## Variational AutoEncoders
In Variational AutoEncoders, we have an additional loss function that forces the innermost layer to follow a unit gaussian distribution. The loss function that gets created is the KL Divergence of the innermost layer with the unit gaussian curve. We also create two innermost layers, once for mean and the other for deviation and then sample that to create the first decoder layer

https://arxiv.org/pdf/1606.05908.pdf

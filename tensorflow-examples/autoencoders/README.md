# AutoEncoders

AutoEncoders are one of the simplest and powerful tools in neural networks. I have tried to provide examples of various kinds of autoencoders

## Vanilla AutoEncoder
In Vanilla AutoEncoders, we simply have the same data as input and output and fully connected layers

## Cross Entropy AutoEncoder

## Denoising AutoEncoder

## Tied Weight AutoEncoder

## Variational AutoEncoder
In Variational AutoEncoders, we have an additional loss function that forces the innermost layer to follow a unit gaussian distribution. The loss function that gets created is the KL Divergence of the innermost layer with the unit gaussian curve. We also create two innermost layers, once for mean and the other for deviation and then sample that to create the first decoder layer

https://arxiv.org/pdf/1606.05908.pdf

## Sparse AutoEncoder

This is based on the following paper https://web.stanford.edu/class/cs294a/sparseAutoencoder_2011new.pdf

## Sparse AutoEncoder With graphs
It is the same as above, but added graphs in order to be able to analyse

## WORD2VEC

This is an implementation of logic mentioned in the following article
https://towardsdatascience.com/learn-word2vec-by-implementing-it-in-tensorflow-45641adaf2ac

### Concerns
#### 1) Convergence Issues
We need to tune the model in order to be able to bypass the convergence issues
#### 2) Accuracy
We need some metrics to be able to find out whether the model output is good
#### 3) Other methods apart from SkipGram
The following paper has more information that we need to implement
https://arxiv.org/pdf/1301.3781.pdf

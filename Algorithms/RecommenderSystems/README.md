# Recommender Systems 
Recommender Systems are the backbone of any transactional selling model

## CollaborativeFilteringSVD
This performs Collaborative Filtering on the SVD results for an input data

## VanillaRBMRecommender
[Restricted Boltzmann Machines](https://en.wikipedia.org/wiki/Restricted_Boltzmann_machine) are simple yet powerful tools for mimicking gradient descent learning. Here we see how we can use this for coming up with recommendations


This is the [Restricted Boltzmann Machines for Collaborative Filtering paper](https://www.cs.toronto.edu/~rsalakhu/papers/rbmcf.pdf)
It covers Unit 2 within the paper. Other units are work in Progress

Tere are 2 methodologies mentioned here
 - Normal Gibbs Sampling with cutoff on 0.5
 - Bernoulli Sampling during each pass of Gibbs ( Recommended )

## RBMWithStoppingCriterion
This is based on the following paper<br/>
 * https://www.researchgate.net/publication/280498168_A_Neighbourhood-Based_Stopping_Criterion_for_Contrastive_Divergence_Learning<br/>
 * https://arxiv.org/abs/1507.06803

## Non Negative Matrix Factorization
This is a powerful method if we want to just find the similar users and similar products from a user item matrix
It is based on an old paper
 * https://papers.nips.cc/paper/1861-algorithms-for-non-negative-matrix-factorization.pdf

## Content Based Filtering
Content Based filtering is a simple tool to solve the cold start problems that most of the Recommender systems face
Some resources which were inspirations for the code are
 * https://www.analyticsvidhya.com/blog/2015/08/beginners-guide-learn-content-based-recommender-systems/
 * http://recommender-systems.org/content-based-filtering/

## Probabilistic Matrix Factorization ( Under Development )
This is based on the paper <br/>
https://papers.nips.cc/paper/3208-probabilistic-matrix-factorization.pdf




## TO DO
 - Coding up https://www.cs.toronto.edu/~rsalakhu/papers/rbmcf.pdf

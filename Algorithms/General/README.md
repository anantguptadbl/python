# General Algorithms

This page contains some non-categorised algorithms that are useful in day to day analysis

## Gaussian Mixture Model
https://mdav.ece.gatech.edu/ece-6254-spring2017/notes/20-GMMs-EM.pdf  
The above paper considers the entire covariance matrix while calculating the probabilities

However, instead of taking the entire covariance matrix, i have taken only diagonal values

## Twiddle Algorithm
https://martin-thoma.com/twiddle/

## CLUSTERING ELBOW CURVE
For a Centroid based clustering algorithm, we will have to finalize the optimal number of clusters
Currently the way we find the optimal number of clusters is by analyzing the elbow curve. My attempt is to automate that process for efficiency. I have made use of polyfit function in python
</br>
https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.polyfit.html

#### TODO
http://www.cs.rpi.edu/~magdon/ps/journal/LowRank_IJDMM.pdf  
This is another implementation of an approximation of the covariance matrix



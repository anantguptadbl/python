# Simple Collective Matrix Factorization ( Work in Progress )

# Collective Matrix Factorization

import numpy as np
import pandas as pd
import tensorflow as tf
import scipy

# Initialization
epochs=100000
numUsers=6
numItems=6
userFeatures=4
itemFeatures=4
userEmbeddings=3
itemEmbeddings=3
UI=tf.placeholder(tf.float32,shape=(numUsers,numItems)) # User Item Data
UF=tf.placeholder(tf.float32,shape=(numUsers,userFeatures)) # User Features
IF=tf.placeholder(tf.float32,shape=(numItems,itemFeatures)) # Item Features

# Variables
UE=tf.Variable(tf.random_uniform([numUsers,userEmbeddings])) # User Embeddings
IE=tf.Variable(tf.random_uniform([numItems,itemEmbeddings])) # Item Embeddings
UFE=tf.Variable(tf.random_uniform([userFeatures,userEmbeddings])) # User Feature Embeddings
IFE=tf.Variable(tf.random_uniform([userFeatures,itemEmbeddings])) # Item Feature Embeddings

# Loss Function
loss=tf.reduce_mean(tf.pow(UI - tf.matmul(UE,tf.transpose(IE)) ,2)) + tf.reduce_mean(tf.pow(UF - tf.matmul(UE,tf.transpose(UFE)),2)) + tf.reduce_mean(tf.pow(IF - tf.matmul(IE,tf.transpose(IFE)),2))

# Training
train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# Start the training
sess=tf.Session()
init=tf.global_variables_initializer()
sess.run(init)

# Input Data
userItemData=np.array([[0,1,1,1,0,0],[1,1,1,0,0,0],[0,1,0,0,0,1],[1,1,0,0,0,0],[0,0,0,1,1,1],[1,1,1,1,0,0]])
userFeaturesData=np.array([[1,2,3,1],[1,1,4,3],[1,1,1,1],[0,1,1,0],[1,0,0,1],[0,2,0,2]])
itemFeaturesData=np.array([[1,2,3,1],[1,1,4,3],[1,1,1,1],[0,1,1,0],[1,0,0,1],[0,2,0,2]])

for curEpoch in range(epochs):
    _,curLoss=sess.run([train_step,loss], feed_dict={UI:userItemData,UF:userFeaturesData,IF:itemFeaturesData})
    if(curEpoch % 10000==0):
        print("curEpoch {} = {}".format(curEpoch,curLoss))
userEmbeddings,itemEmbeddings=sess.run([UE,IE], feed_dict={UI:userItemData,UF:userFeaturesData,IF:itemFeaturesData})

# Testing
import scipy
scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(userEmbeddings))


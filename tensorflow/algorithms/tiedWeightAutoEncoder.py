# Tied Weight AutoEncoder
# This autoencoder contains the the same weight matrix for the encoder and decoder sections

import tensorflow as tf
import numpy as np

class TFTiedAutoEncoder():
    # INIT function
    def __init__(self,X,learningRate):
        # Input Data
        self.X=X
        
        # Input Placeholder
        self.x_input=tf.placeholder("float32",(None,X.shape[1]))
        
        # Intermediate Variables
        # Weights
        self.encoder_1_weight=tf.Variable(tf.random_uniform([self.X.shape[1],self.X.shape[1]/2]))
        self.encoder_2_weight=tf.Variable(tf.random_uniform([self.X.shape[1]/2,self.X.shape[1]/3]))
        # Biases
        self.encoder_1_bias=tf.Variable(tf.random_uniform([self.X.shape[1]/2]))
        self.encoder_2_bias=tf.Variable(tf.random_uniform([self.X.shape[1]/3]))
        self.decoder_1_bias=tf.Variable(tf.random_uniform([self.X.shape[1]/2]))
        self.decoder_2_bias=tf.Variable(tf.random_uniform([self.X.shape[1]]))
        
        # Setting up the layer interactions        
        self.encoder_1=tf.add(tf.matmul(self.x_input,self.encoder_1_weight),self.encoder_1_bias)
        self.encoder_2=tf.nn.sigmoid(tf.add(tf.matmul(self.encoder_1,self.encoder_2_weight),self.encoder_2_bias))
        self.decoder_1=tf.nn.sigmoid(tf.add(tf.matmul(self.encoder_2,tf.transpose(self.encoder_2_weight)),self.decoder_1_bias))
        self.decoder_2=tf.add(tf.matmul(self.decoder_1,tf.transpose(self.encoder_1_weight)),self.decoder_2_bias)
        
        # The loss and optimizer functions
        self.loss=tf.reduce_mean(tf.pow(self.x_input-self.decoder_2,2))
        self.optimizer=tf.train.GradientDescentOptimizer(learningRate).minimize(self.loss)
        self.init=tf.global_variables_initializer()
        
    def train(self,execRange=1000):
        with tf.Session() as sess:
            sess.run(self.init)
            for curIteration in range(execRange):
                _,curLoss=sess.run([self.optimizer,self.loss],feed_dict={self.x_input:self.X})
                if(curIteration % 100==0):
                    print("The loss at step {} is {}".format(curIteration,curLoss))
                
if __name__=="__main__":
    # Input Data
    data=np.random.rand(1000,5)
    
    # Arg1 : The data
    # Arg2 : The learning rate
    ae=TFTiedAutoEncoder(data,0.1)
    
    # Arg1 : The number of iterations of training
    ae.train(5000)
        
        


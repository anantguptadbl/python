# Denoising Autoencoder
#http://www.iro.umontreal.ca/~lisa/publications2/index.php/attachments/single/176

# Autoencoder example
import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt

class TFDenoisingAutoEncoder():
    # INIT function
    # Arg1 : The data
    # Arg2 : The learning rate
    def __init__(self,X,learningRate):
        # Input Data
        self.X=X
        
        # Input Placeholder
        self.x_input=tf.placeholder("float32",(None,X.shape[1]))
        
        # Intermediate Variables
        # Weights
        self.encoder_1_weight=tf.Variable(tf.random_uniform([self.X.shape[1],self.X.shape[1]/2]))
        self.encoder_2_weight=tf.Variable(tf.random_uniform([self.X.shape[1]/2,self.X.shape[1]/3]))
        self.decoder_1_weight=tf.Variable(tf.random_uniform([self.X.shape[1]/3,self.X.shape[1]/2]))
        self.decoder_2_weight=tf.Variable(tf.random_uniform([self.X.shape[1]/2,self.X.shape[1]]))
        # Biases
        self.encoder_1_bias=tf.Variable(tf.random_uniform([self.X.shape[1]/2]))
        self.encoder_2_bias=tf.Variable(tf.random_uniform([self.X.shape[1]/3]))
        self.decoder_1_bias=tf.Variable(tf.random_uniform([self.X.shape[1]/2]))
        self.decoder_2_bias=tf.Variable(tf.random_uniform([self.X.shape[1]]))
        
        # Intermediate Layer Operations
        self.encoder_1=tf.add(tf.matmul(self.x_input,self.encoder_1_weight),self.encoder_1_bias)
        self.encoder_2=tf.nn.sigmoid(tf.add(tf.matmul(self.encoder_1,self.encoder_2_weight),self.encoder_2_bias))
        self.decoder_1=tf.nn.sigmoid(tf.add(tf.matmul(self.encoder_2,self.decoder_1_weight),self.decoder_1_bias))
        self.decoder_2=tf.add(tf.matmul(self.decoder_1,self.decoder_2_weight),self.decoder_2_bias)
        
        # Loss and Optimizer
        self.loss=tf.reduce_mean(tf.pow(self.x_input-self.decoder_2,2))
        self.optimizer=tf.train.GradientDescentOptimizer(learningRate).minimize(self.loss)
        self.init=tf.global_variables_initializer()
    # Arg1 : Pass the corruption rate. The ratio of the input data will be suppressed to 0
    def corruptTheInput(self,corruptionRate):
        RowIndexes=np.array(random.sample(range(0,self.X.shape[0]), int(1.0 * corruptionRate * self.X.shape[0])))
        self.X[RowIndexes,:]=0
        
    # Arg1 : Pass the corruption rate. The ratio of the input rows and columns will be suppressed to 0
    def corruptTheInputPartially(self,corruptionRate):
        RowIndexes=np.array(random.sample(range(0,self.X.shape[0]), int(1.0 * corruptionRate * self.X.shape[0])))
        FeatureIndexes=np.array(random.sample(range(0,self.X.shape[1]),int(1.0 * corruptionRate * self.X.shape[1])))
        self.X[RowIndexes[:,None],FeatureIndexes]=0
    
    def train(self,execRange=1000):
        self.LossArray=[]
        with tf.Session() as sess:
            sess.run(self.init)
            for curIteration in range(execRange):
                _,curLoss=sess.run([self.optimizer,self.loss],feed_dict={self.x_input:self.X})
                self.LossArray.append(curLoss)
                if(curIteration % 1000==0):
                    print("The loss at step {} is {}".format(curIteration,curLoss))
                
if __name__=="__main__":
    # Input Data
    data=np.random.rand(100,6)
    
    
    ae=TFDenoisingAutoEncoder(data,0.1)
    ae.corruptTheInput(0.2)
    ae.train(5000)
    
    # We will now corrupt the features partially
    ae1=TFDenoisingAutoEncoder(data,0.1)
    ae1.corruptTheInputPartially(0.2)
    ae1.train(5000)
    
    # We will now not do any corruption
    ae2=TFDenoisingAutoEncoder(data,0.1)
    ae2.train(5000)
    
    plt.figure(figsize=(20,10))
    plt.plot(range(len(ae2.LossArray)),ae2.LossArray,label='NoNoise',color='red')
    plt.plot(range(len(ae1.LossArray)),ae1.LossArray,label='FullRowNoise',color='blue')
    plt.plot(range(len(ae.LossArray)),ae.LossArray,label='PartialRowNoise',color='green')
    plt.legend()
    plt.show()

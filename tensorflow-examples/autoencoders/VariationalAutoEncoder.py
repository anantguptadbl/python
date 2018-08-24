# Autoencoder example
import tensorflow as tf
import numpy as np

class TFVariationalAutoencoder():
    # INIT function
    def __init__(self,X,learningRate):
        # Input Data
        self.X=X
        
        # Input Placeholder
        self.x_input=tf.placeholder("float32",(None,X.shape[1]))
        
        # Intermediate Variables
        self.encoder_1_weight=tf.Variable(tf.random_uniform([self.X.shape[1],self.X.shape[1]/2]))
        self.encoder_2_mean_weight=tf.Variable(tf.random_uniform([self.X.shape[1]/2,self.X.shape[1]/3]))
        self.encoder_2_deviation_weight=tf.Variable(tf.random_uniform([self.X.shape[1]/2,self.X.shape[1]/3]))
        self.decoder_1_weight=tf.Variable(tf.random_uniform([self.X.shape[1]/3,self.X.shape[1]/2]))
        self.decoder_2_weight=tf.Variable(tf.random_uniform([self.X.shape[1]/2,self.X.shape[1]]))
        
        self.encoder_1_bias=tf.Variable(tf.random_uniform([self.X.shape[1]/2]))
        self.encoder_2_mean_bias=tf.Variable(tf.random_uniform([self.X.shape[1]/3]))
        self.encoder_2_deviation_bias=tf.Variable(tf.random_uniform([self.X.shape[1]/3]))
        self.decoder_1_bias=tf.Variable(tf.random_uniform([self.X.shape[1]/2]))
        self.decoder_2_bias=tf.Variable(tf.random_uniform([self.X.shape[1]]))
                
        self.encoder_1=tf.add(tf.matmul(self.x_input,self.encoder_1_weight),self.encoder_1_bias)
        
        # The second encoder layer will be two layers. One denoting mean and the other standard deviation
        self.encoder_2_mean=tf.nn.sigmoid(tf.add(tf.matmul(self.encoder_1,self.encoder_2_mean_weight),self.encoder_2_mean_bias))
        self.encoder_2_deviation=tf.nn.sigmoid(tf.add(tf.matmul(self.encoder_1,self.encoder_2_deviation_weight),self.encoder_2_deviation_bias))
        
        samples=tf.random_normal([self.X.shape[0],self.X.shape[1]/2],0,1,dtype=tf.float32)
        self.decoder_1=self.encoder_2_mean + ( self.encoder_2_deviation * samples)
        self.decoder_2=tf.add(tf.matmul(self.decoder_1,self.decoder_2_weight),self.decoder_2_bias)
        
        # We will have two kinds of losses. One the normal recreationLoss between input and ouput and the other
        # How much is the sampling of mean and stddev layer resulting in a normal distribution
        self.recreationLoss=tf.reduce_mean(tf.pow(self.x_input-self.decoder_2,2)) 
        self.KLLoss=-0.5 * tf.reduce_sum(1 + self.encoder_2_deviation - tf.square(self.encoder_2_mean) - tf.exp(self.encoder_2_deviation),axis=1)
        self.loss=tf.reduce_mean(self.recreationLoss + self.KLLoss)
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
    ae=TFVariationalAutoencoder(data,0.1)
    
    # Arg1 : The number of iterations of training
    ae.train(50000)

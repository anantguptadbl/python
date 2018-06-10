# Cross Entropy Autoencoder example
import tensorflow as tf
import numpy as np

class TFAutoEncoderCE():
    # INIT function
    def __init__(self,X,learningRate):
        # Input Data
        self.X=X
        
        # Input Placeholder
        self.x_input=tf.placeholder("float32",(None,X.shape[1]))
        
        # Intermediate Variables
        # Weights
        self.encoder_1_weight=tf.Variable(tf.random_uniform([self.X.shape[1],self.X.shape[1]/2]))
        self.decoder_1_weight=tf.Variable(tf.random_uniform([self.X.shape[1]/2,self.X.shape[1]]))
        # Biases
        self.encoder_1_bias=tf.Variable(tf.random_uniform([self.X.shape[1]/2]))
        self.decoder_1_bias=tf.Variable(tf.random_uniform([self.X.shape[1]]))
        
        # Encoding Layer Operations        
        self.encoder_1=tf.nn.sigmoid(tf.add(tf.matmul(self.x_input,self.encoder_1_weight),self.encoder_1_bias))
        self.decoder_1=tf.add(tf.matmul(self.encoder_1,self.decoder_1_weight),self.decoder_1_bias)
        
        # Loss and Optimizer
	# Cross Entropy Loss
        self.loss=tf.reduce_sum((self.x_input * tf.log(self.decoder_1)) + (1 - self.x_input)*tf.log(1-self.decoder_1))
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
    data=np.random.rand(100,4)
    
    # Arg1 : The data
    # Arg2 : The learning rate
    ae=TFAutoEncoderCE(data,0.1)
    
    # Arg1 : The number of iterations of training
    ae.train(5000)

# Linear Regression on Tensorflow

# Imports
import tensorflow as tf
import numpy as np


class TFLinearRegression():
    
    def __init__(self,X,Y):    
        self.X=X
        self.Y=Y
        # Setting the placeholders which will be inputs
        self.x=tf.placeholder("float32",(None,X.shape[1]))
        self.y=tf.placeholder("float32",(None,1))

        # Setting up the Variables
        self.w=tf.Variable(tf.random_uniform([NumberOfFeatures,1]))
        self.b=tf.Variable(tf.random_uniform([1]))

        # Output
        self.output=tf.add(tf.matmul(self.x,self.w),self.b)

        # Optimizer
        self.loss=tf.reduce_mean(tf.pow(self.output - self.y,2))
        self.optimizer=tf.train.GradientDescentOptimizer(0.1).minimize(self.loss)
        self.init=tf.global_variables_initializer()
        

    def train(self):
        # Train of Tensorflow
        # Run the session
        with tf.Session() as sess:
            sess.run(init)
            for runCounter in range(10000):
                _,l=sess.run([optimizer,loss],feed_dict={x:self.X,y:self.Y.reshape(self.X.shape[0],1)})
                if(runCounter % 100==0):
                    print("The loss after {} is {}".format(runCounter,l))
            weights,bias=sess.run([w,b],feed_dict={x:self.X,y:self.Y.reshape(self.X.shape[0],1)})    
        return({"weights":weights,"bias":bias})

if __name__=="__main__":
   
    # Read/Prepare the data
    data=np.random.rand(100,5)
    
    lr=TFLinearRegression(data[:,0:4],data[:,4])
    results=lr.train()
    print(results)

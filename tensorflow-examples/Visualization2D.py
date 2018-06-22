# How to plot high dmensional data

from sklearn.manifold import TSNE
import tensorflow as tf
import numpy as np

class Feature2DVisualization():
    def __init__(self,X,learningRate=0.1):
        self.X=X
        self.x_input=tf.placeholder("float32",(None,X.shape[1]))
        self.w=tf.Variable(tf.random_uniform([X.shape[1],2]))
        self.b=tf.Variable(tf.random_uniform([2]))

        self.w1=tf.Variable(tf.random_uniform([2,X.shape[1]]))
        self.b1=tf.Variable(tf.random_uniform([X.shape[1]]))

        self.innerLayer=tf.add(tf.matmul(self.x_input,self.w),self.b)
        self.output=tf.add(tf.matmul(self.innerLayer,self.w1),self.b1)

        self.loss=tf.reduce_mean(tf.pow(self.output-self.x_input,2))
        self.optimizer=tf.train.GradientDescentOptimizer(learningRate).minimize(self.loss)

        self.init=tf.global_variables_initializer()

    def train(self):
        with tf.Session() as sess:
            sess.run(self.init)
            for x in range(100):
                _,l=sess.run([self.optimizer,self.loss],feed_dict={self.x_input:self.X})
                print("the loss is {}".format(l))
            self.weight,self.bias=sess.run([self.w,self.b],feed_dict={self.x_input:self.X})
        return([self.weight,self.bias])
    
    def TSNEComparison(self):
        # Comparison with TSNE
        print("Running TSNE")
        inputEmbedded=TSNE(n_components=2).fit_transform(self.X)
        print("Recreating the matrix from Neural Network")
        inputEmbeddedNN=np.add(np.matmul(self.X,self.weight),self.bias)
        print("Plotting the two datasets")
        import matplotlib.pyplot as plt
        plt.scatter(inputEmbedded[:,0],inputEmbedded[:,1],color='red',label='TSNE',alpha=0.2)
        plt.legend()
        plt.show()

        plt.figure()
        plt.scatter(inputEmbeddedNN[:,0],inputEmbeddedNN[:,1],color='blue',label='NN',alpha=0.2)
        plt.legend()
        plt.show()
        
if __name__=="__main__":
   
    # Read/Prepare the data
    # Sample Data
    data=np.random.rand(1000,5)
    
    viz2D=Feature2DVisualization(data,0.1)
    weight,bias=viz2D.train()
    viz2D.TSNEComparison()

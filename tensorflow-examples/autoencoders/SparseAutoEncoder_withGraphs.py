# Autoencoder example
import tensorflow as tf
import numpy as np

class TFSparseAutoEncoderGraph():
    # INIT function
    def __init__(self,X,learningRate,alpha,beta,klThreshold):
        # Input Data
        self.X=X
        self.alpha=alpha
        self.beta=beta
        self.klThreshold=klThreshold
        
        
        
        # Input Placeholder
        self.x_input=tf.placeholder("float32",(None,X.shape[1]))
        
        # Intermediate Variables
        self.encoder_1_weight=tf.Variable(tf.random_uniform([self.X.shape[1],self.X.shape[1]/2]),name="encoder1Weight")
        self.encoder_2_weight=tf.Variable(tf.random_uniform([self.X.shape[1]/2,self.X.shape[1]/3]),name="encoder2Weight")
        self.decoder_1_weight=tf.Variable(tf.random_uniform([self.X.shape[1]/3,self.X.shape[1]/2]),name="decoder1Weight")
        self.decoder_2_weight=tf.Variable(tf.random_uniform([self.X.shape[1]/2,self.X.shape[1]]),name="decoder2Weight")
        self.encoder_1_bias=tf.Variable(tf.random_uniform([self.X.shape[1]/2]),name="encoder1Bias")
        self.encoder_2_bias=tf.Variable(tf.random_uniform([self.X.shape[1]/3]),name="encoder2Bias")
        self.decoder_1_bias=tf.Variable(tf.random_uniform([self.X.shape[1]/2]),name="decoder1Bias")
        self.decoder_2_bias=tf.Variable(tf.random_uniform([self.X.shape[1]]),name="decoder2Bias")
        
        # Computed Layers
        self.encoder_1=tf.add(tf.matmul(self.x_input,self.encoder_1_weight),self.encoder_1_bias)
        self.encoder_2=tf.nn.sigmoid(tf.add(tf.matmul(self.encoder_1,self.encoder_2_weight),self.encoder_2_bias))
        self.decoder_1=tf.nn.sigmoid(tf.add(tf.matmul(self.encoder_2,self.decoder_1_weight),self.decoder_1_bias))
        self.decoder_2=tf.add(tf.matmul(self.decoder_1,self.decoder_2_weight),self.decoder_2_bias)
        
        # For sparse Autoencoder, the loss function will vary a bit
        # We want the mean of activations for the self.encoder_2 layer to be very low, so that in sparsity only few neurons
        # are activated. Most of the neurons are 0
        self.encodedMeans=tf.reduce_mean(self.encoder_2,axis=0)
        self.KLDivergenceVal=self.KLDivergence(self.klThreshold,self.encodedMeans)
        
        self.loss= 0.5*tf.reduce_mean(tf.pow(self.x_input-self.decoder_2,2)) \
                   + 0.5 * self.alpha * \
                                    (tf.nn.l2_loss(self.encoder_1_weight) +  \
                                     tf.nn.l2_loss(self.encoder_2_weight) +  \
                                     tf.nn.l2_loss(self.decoder_1_weight) +  \
                                     tf.nn.l2_loss(self.decoder_2_weight) \
                                    )   \
                   + self.beta * tf.reduce_sum(self.KLDivergenceVal)
            
        tf_loss_summary = tf.summary.scalar('loss', self.loss)
        #tf_KLDivergence_summary = tf.summary.scalar('KLDivergence', self.KLDivergenceVal)
        tf_encoder1loss_summary = tf.summary.scalar('encoder_1_loss', tf.nn.l2_loss(self.encoder_1_weight))
        tf_encoder2loss_summary = tf.summary.scalar('encoder_2_loss', tf.nn.l2_loss(self.encoder_2_weight))
        tf_decoder1loss_summary = tf.summary.scalar('decoder_1_loss', tf.nn.l2_loss(self.decoder_1_weight))
        tf_decoder2loss_summary = tf.summary.scalar('decoder_2_loss', tf.nn.l2_loss(self.decoder_2_weight))
        
        self.optimizer=tf.train.GradientDescentOptimizer(learningRate).minimize(self.loss)
        self.init=tf.global_variables_initializer()
        
        # Merge all summaries together
        self.all_summaries = tf.summary.merge([tf_loss_summary,tf_encoder1loss_summary,
                                              tf_encoder2loss_summary,tf_decoder1loss_summary,tf_decoder2loss_summary])
        
    def KLDivergence(self, a, b):
        return (a * tf.log(a) + (1-a)*tf.log(1-a)) - ( a * tf.log(b) + (1-a) * tf.log(1-b))

    def train(self,execRange=1000):
        with tf.Session() as sess:
            sess.run(self.init)
            summ_writer = tf.summary.FileWriter('/home/anantgupta/Documents/Competitions/HackerEarth/tagRecommender', sess.graph)
            for curIteration in range(execRange):
                _,curLoss,lossSummary=sess.run([self.optimizer,self.loss,self.all_summaries],feed_dict={self.x_input:self.X})
                summ_writer.add_summary(lossSummary, curIteration)
                if(curIteration % 100==0):
                    print("The loss at step {} is {}".format(curIteration,curLoss))
                

#data=np.random.rand(1000,5)                
if __name__=="__main__":
    # Input Data

    
    # Arg1 : The data
    # Arg2 : The learning rate
    # Arg3 : This is the weightage given to the Regularization of the weights. This will ensure to what level we have
    # the weights getting activated for each row
    # Arg4 : This is the weightage given to the deviation of the sum of the hidden layer from the KL Threshold
    # Arg5 : This is the mean activations that the innermost layer should have
    # A low value tending towards 0 ensures that most of the activiations are 0
    # A high value tending towards 1 ensures that most of the activations are 1
    ae=TFSparseAutoEncoderGraph(data,learningRate=0.1,alpha=0.1,beta=2,klThreshold=0.1)
    
    # Arg1 : The number of iterations of training
ae.train(1000)

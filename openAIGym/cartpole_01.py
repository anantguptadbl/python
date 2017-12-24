import tensorflow as tf
import pandas as pd
import numpy as np
import os
import random
import gym
import math

class cartPoleSolver1Hidden:
    def __init__(self,savePath):
        # Initialization
        self.num_nodes=8 # Number of nodes in the hidden layer
        self.numLabels=1 # There is just 1 and 0 which will be handled by a single sigmoid function
        self.reverseRow=3 # There is just 1 and 0 which will be handled by a single sigmoid function
        self.numFeatures=4 * self.reverseRow # Number of input features
        self.savePath=savePath # The path where the model will be saved and read and saved again .. inception
        

        # TF Initialization of input features
        self.X=tf.placeholder(tf.float32, [None, self.numFeatures],name='x')
        self.y=tf.placeholder(tf.float32, [None, self.numLabels],name='y')

        # Input and Hidden Layer
        self.weights_hidden = tf.Variable(tf.random_normal([self.numFeatures,self.num_nodes]),name='weights_hidden')
        self.bias_hidden = tf.Variable(tf.random_normal([self.num_nodes]),name='bias_hidden')
        self.preactivations_hidden = tf.add(tf.matmul(self.X, self.weights_hidden), self.bias_hidden)
        self.activations_hidden = tf.nn.sigmoid(self.preactivations_hidden)

        # output layer
        self.weights_output = tf.Variable(tf.random_normal([self.num_nodes, self.numLabels]),name='weights_output')
        self.bias_output = tf.Variable(tf.random_normal([self.numLabels]),name='bias_output') 
        self.preactivations_output = tf.add(tf.matmul(self.activations_hidden, self.weights_output), self.bias_output)
        self.activations_output = tf.nn.sigmoid(self.preactivations_output)

        # Cost function (Mean Squeared Error)
        #cost_OP = tf.nn.l2_loss(preactivations_output-y, name="squared_error_cost")
        self.cost_OP = tf.nn.l2_loss(self.activations_output-self.y, name="squared_error_cost")
        # Optimization Algorithm (Gradient Descent)
        self.learningRate=0.4
        self.training_OP = tf.train.GradientDescentOptimizer(self.learningRate).minimize(self.cost_OP)

    def create_tensorflow_session(self):
        # Create a tensorflow session
        self.sess = tf.Session()
        init_OP = tf.initialize_all_variables()
        self.sess.run(init_OP)
        
    def trainModel(self,inputArray):
        # Initialize all tensorflow variables
        self.prepareData(inputArray)
        self.sess.run(self.training_OP, feed_dict={self.X: self.trainX, self.y: self.trainY})
   
    def reTrainModel(self,inputArray):
        # Initialize all tensorflow variables
        tf.reset_default_graph() 
        self.weights_hidden = tf.get_variable("weights_hidden", shape=[20,self.num_nodes])
        self.bias_hidden = tf.get_variable("bias_hidden", shape=[self.num_nodes])
        self.weights_output = tf.get_variable("weights_output", shape=[self.num_nodes,self.numLabels])
        self.bias_output = tf.get_variable("bias_output", shape=[self.numLabels])
        self.prepareData(inputArray)
        [training,cost]=self.sess.run([self.training_OP,self.cost_OP], feed_dict={self.X: self.trainX, self.y: self.trainY})
        print("The cost is {}".format(cost))

    def saveModel(self):
        # Save the session
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()
        print("Model saved in file: %s" % saver.save(self.sess, self.savePath))
    
    def predict(self,inputArray):
        self.prepareData(inputArray)
        return(self.sess.run(self.activations_output,feed_dict={self.X:self.trainX}))
        
    def prepareData(self,inputDataArray):
        df=pd.DataFrame(inputDataArray,columns=['a1','a2','a3','a4','action'])
        for colName in ['a1','a2','a3','a4']:
            for shiftVal in range(1,self.reverseRow):
                df[colName+'_'+str(shiftVal)]=df[colName].shift(shiftVal)
        # I will categorise the last 5 rows as 0 and others as 1
        #df['action']=[0 if random.random()*10 < 5 else 1 for x in range(df.shape[0])]
        # We will reverse the last reverseRow actions
        df['action']=[0 if x==1 and i > df.shape[0]- self.reverseRow else 1 if x==0 and i > df.shape[0] -self.reverseRow else x for i,x in enumerate(df['action'].values)]
        df.fillna(0,inplace=True)
        # Prepare the trainX and trainY columns
        self.trainX=df[[x for x  in df.columns.values if x !='action']].values
        self.trainY=df['action'].values.reshape((df.shape[0],1))


# GYM Configuration
env = gym.make('CartPole-v0')
env.monitor.start('/tmp/cartpole-experiment-1',force=True)
#env = gym.wrappers.Monitor(env, '/tmp/cartpole-experiment-2')
#env.monitor.start('/tmp/cartpole-experiment-2',force=True)


# Initial Training
savePath='/var/tmp/cartPole01Model/cartPoleModel'
model1=cartPoleSolver1Hidden(savePath)
model1.create_tensorflow_session()
import random
a=[[ 0.97561885,  0.82538392,  1.00646574,  1.30465421,1], 
[ 0.97212653,  0.63017043,  1.01255882,  1.59936925,0], 
[ 0.96472993,  0.43487504,  1.02454621,  1.89598143,1], 
[ 0.95342743,  0.23942904,  1.04246584,  2.19627798,1], 
[ 0.93821602,  0.04378387,  1.0663914 ,  2.50196236,0], 
[ 0.91909169, -0.15207835,  1.09643065,  2.81461317,0], 
[ 0.89605013, -0.34813241,  1.13272291,  3.1356347 ,0], 
[ 0.86908748, -0.54429533,  1.1754356 ,  3.46619734,0], 
[ 0.83820157, -0.74040917,  1.22475955,  3.80716665,0]]
model1.trainModel(a)


lastMax=0

# Latest changes
for i_episode in range(100):
	# Data for training
	trainingData=[]
	env.reset()
	best_cs = (np.random.rand(4) * 2 - 1)
	observation = env.reset()
	env.render()
	action = env.action_space.sample()
	tries=0
	done=0
	while not done:
		observation, reward, done, info = env.step(action)
		trainingData.append(list(observation) + list([action]))
		# Predicting the next action
		predictions=model1.predict(trainingData[-5:])
		if(predictions[-1:] > 0.7):
			action=1
		else:
			action=0
		#print("Reward: {} Observation: {} Done: {} Action:{}".format(reward,observation,done,action))
		tries=tries+1
		if done:
			print("Episode finished after {} timesteps".format(tries+1))
	if(len(trainingData)>10):
		lastMax=len(trainingData)
		print("The number of rows into the trainingData is {}".format(len(trainingData)))
		model1.reTrainModel(trainingData)

# Save the model
#model1.saveModel()


env.monitor.close()

# Uncomment this when you want to upload your file

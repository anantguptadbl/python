# Vanilla RNN Class

# Imports
import tensorflow as tf
import pandas as pd
import numpy as np

class VanillaRNN():
    def __init__(self,X,Y,sess):
        # CONFIGURATIONS
        self.BATCH_SIZE=1
        self.NUM_STEPS=5
        self.PARTITION_SIZE=X.shape[0] // self.BATCH_SIZE
        self.EPOCH_SIZE=self.PARTITION_SIZE // self.NUM_STEPS
        self.STATE_SIZE=2
        self.NUM_CLASSES=5
        self.NUM_CLASSES_OUTPUT=2
        self.LEARNING_RATE=0.1
        self.EPOCHRUNS=100
        self.X=X
        self.Y=Y
        self.sess=sess
        self.x=tf.placeholder(tf.int32,[None,self.NUM_STEPS])
        self.y=tf.placeholder(tf.int32,[None,self.NUM_STEPS])
        self.init_state=tf.zeros([self.BATCH_SIZE,self.STATE_SIZE])
        self.x_one_hot = tf.one_hot(self.x, self.NUM_CLASSES)
        self.rnn_inputs = tf.unstack(self.x_one_hot, axis=1)
        self.W_rnnCell = tf.Variable(np.random.rand(self.NUM_CLASSES + self.STATE_SIZE, self.STATE_SIZE), dtype=tf.float32,name='W_rnnCell')
        self.b_rnnCell = tf.Variable(np.zeros(self.STATE_SIZE), dtype=tf.float32,name='b_rnnCell')
        self.W_softmax = tf.Variable(np.random.rand(self.STATE_SIZE, self.NUM_CLASSES_OUTPUT),dtype=tf.float32,name='W_softmax')
        self.b_softmax = tf.Variable(np.zeros(self.NUM_CLASSES_OUTPUT),dtype=tf.float32,name='b_softmax')

    
    def keepProvidingDatachunks(self):
        # Create the feed data
        data_input_x=np.array([self.X[self.PARTITION_SIZE*curBatch:self.PARTITION_SIZE * (curBatch + 1)] for curBatch in range(self.BATCH_SIZE)])
        data_input_y=np.array([self.Y[self.PARTITION_SIZE*curBatch:self.PARTITION_SIZE * (curBatch + 1)] for curBatch in range(self.BATCH_SIZE)])
        for curCounter in range(self.EPOCH_SIZE):
            x = data_input_x[:, curCounter * self.NUM_STEPS:(curCounter + 1) * self.NUM_STEPS]
            y = data_input_y[:, curCounter * self.NUM_STEPS:(curCounter + 1) * self.NUM_STEPS]
            yield(x,y)


    def addCellsToGraph(self):
    # Add the RNN Cells
        #state = self.init_state
        self.rnn_outputs = []
        rnn_cell = tf.nn.rnn_cell.BasicRNNCell(self.STATE_SIZE)
        outputs, state = tf.nn.dynamic_rnn(rnn_cell, self.x_one_hot,initial_state=self.init_state,dtype=tf.float32)
        self.rnn_outputs=tf.unstack(outputs,num=self.NUM_STEPS,axis=1)
        self.final_state=state

    def setupLossFunctions(self):
    # Setting up the LOSS functions and the Optmizers
        self.logits = [tf.matmul(rnn_output, self.W_softmax) + self.b_softmax for rnn_output in self.rnn_outputs]
        self.predictions = [tf.nn.softmax(logit) for logit in self.logits]
        # Turn our y placeholder into a list of labels
        self.y_as_list = tf.unstack(self.y, num=self.NUM_STEPS, axis=1)
        #losses and train_step
        self.total_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_as_list,logits=self.logits)
        self.train_step = tf.train.AdagradOptimizer(self.LEARNING_RATE).minimize(self.total_loss)

        # Training the network
    def train_network(self):
        self.sess.run(tf.global_variables_initializer())
        self.training_losses = []
        self.training_state = np.zeros((self.BATCH_SIZE,self.STATE_SIZE))
        for nnEpochs in range(self.EPOCHRUNS):
            training_loss = 0
            for idx, (curX,curY) in enumerate(self.keepProvidingDatachunks()):
                training_loss_, training_state, _,logitData,y_as_list_data=self.sess.run([self.total_loss,
                              self.final_state,
                              self.train_step,
                              self.logits,
                            self.y_as_list
                                ],
                                  feed_dict={self.x:curX, self.y:curY, self.init_state:self.training_state})
                training_loss += training_loss_
            self.training_losses.append(training_loss/100)
        return self.training_losses,self.training_state
    
    def test_network(self,testX):
        print(testX)
        return(self.sess.run([self.predictions],feed_dict={self.x:testX}))

    def predicted_labels(self):
        fullResult=[]
        print(self.X.shape[0]//self.NUM_STEPS)
        for counter in range(self.X.shape[0]//self.NUM_STEPS):
            predictedLogit=self.sess.run([self.predictions],feed_dict={self.x:self.X[counter*self.NUM_STEPS : (counter +1) * self.NUM_STEPS].reshape(1,self.NUM_STEPS)})
            #print(len(predictedLogit[0]))
            predictedLabels=[np.argmax(curLogits) for curLogits in predictedLogit[0]]
            fullResult=fullResult  + zip(self.X,self.Y,predictedLabels)
        return(pd.DataFrame(fullResult,columns=['X','Y','PREDICTED_Y']))
    
if __name__=="__main__":
    XData = np.array(np.random.choice(5, size=(100,)))
    YData = np.array(np.random.choice(2, size=(100,)))

    tf.reset_default_graph()

    with tf.Session() as sess:
        vrnn=VanillaRNN(XData,YData,sess)
        vrnn.addCellsToGraph()
        vrnn.setupLossFunctions()
        vrnn.train_network()
        
        # Get the actual and predicted labels
        print(vrnn.predicted_labels())
        
        # Test with a single Array
        print(vrnn.test_network(np.array([4,0,4,0,1]).reshape(1,5)))

# IMPORTS
import tensorflow as tf
import numpy as np
from tensorflow.python.ops.rnn_cell import LSTMCell

# TF LSTM
class LSTMClassification(object):
    def __init__(self,inputData,outputData,hidden_num):
        # This is the data on which the gradients are calculated
        self.batch_num = inputData[0].get_shape().as_list()[0]
        # The encoding of each data point
        self.elem_num = inputData[0].get_shape().as_list()[1]
        # Setting the LSTM Cell
        self.lstmCell=LSTMCell(hidden_num)
        (self.lstmOutput, self.lstmState) = tf.contrib.rnn.static_rnn(self.lstmCell, inputData, dtype=tf.float32)
        self.lstmOutput=tf.stack(self.lstmOutput)
        self.loss = tf.reduce_mean(tf.square(outputData - self.lstmOutput))
        self.train = tf.train.AdamOptimizer().minimize(self.loss)

class LSTMPrediction(object):
    def __init__(self,inputData,LSTMClassificationObject):
        (self.lstmOutput, self.lstmState) = tf.contrib.rnn.static_rnn(LSTMClassificationObject.lstmCell, inputData, dtype=tf.float32)
        
tf.reset_default_graph()
a=np.array([[1,2,9],[8,2,1],[1,7,1],[2,7,3],[2,2,7],[3,4,3],[3,3,3],[9,4,5],[5,6,7],[5,6,4],[5,4,3],[5,6,6]],dtype=np.int32)
b=np.array([[1,0,0],[0,1,0],[1,0,1],[0,0,1],[1,1,0],[1,1,0],[0,0,1],[1,0,0],[0,0,1],[0,1,1],[1,1,0],[0,0,1]],dtype=np.int32)
a=a.reshape(4,3,3)
b=b.reshape(4,3,3)
step_num=4
batch_num=3
elem_num=3
iteration=300
hidden_num=3
p_input = tf.placeholder(tf.float32, shape=(step_num, batch_num, elem_num))
p_output = tf.placeholder(tf.float32, shape=(step_num, batch_num, elem_num))
test_input = tf.placeholder(tf.float32, shape=(1, batch_num, elem_num))
p_inputs=[tf.squeeze(t,0) for t in tf.split(p_input,step_num,0)]
p_outputs=[tf.squeeze(t,0) for t in tf.split(p_output,step_num,0)]
p_test=[tf.squeeze(t,0) for t in tf.split(test_input,1,0)]

#print("Input into the class is {}".format(a.shape))
#print("Batch num calculated is {}".format(p_inputs[0].get_shape().as_list()[0]))
#print("Elem num calculated is {}".format(p_inputs[0].get_shape().as_list()[1]))

lstmModelTrain = LSTMClassification(p_inputs,p_outputs,hidden_num)
lstmModelTest = LSTMPrediction(p_test,lstmModelTrain)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(iteration):
        (loss_val, _) = sess.run([lstmModelTrain.loss, lstmModelTrain.train], {p_input:a,p_output:b})
        if(i % 100==0):
            print('iter %d:' % (i + 1), loss_val)
    
    # NEW PREDICTION
    c=np.array([[1,2,9],[8,2,1],[1,7,5]],dtype=np.int32)
    c=c.reshape(1,3,3)
    resultPredicted=sess.run([lstmModelTest.lstmOutput],{test_input:c})
    print(resultPredicted)
    

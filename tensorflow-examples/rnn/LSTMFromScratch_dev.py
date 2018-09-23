# LSTM from scratch
import numpy as np
import tensorflow as tf

# Global configuration
step_num=5
hidden_num=12
elem_num=1

# Input Placeholders
x=tf.placeholder(shape=[None,step_num,elem_num],dtype=tf.int32)
y=tf.placeholder(shape=[None,step_num,elem_num],dtype=tf.int32)

# Configuration and Vairables
CellState=tf.Variable(tf.truncated_normal([step_num,step_num]),dtype=tf.float32)
HiddenState=tf.Variable(tf.truncated_normal([step_num,hidden_num]),dtype=tf.float32)

LSTMLoopInitState=tf.placeholder(shape=[2,None,step_num], dtype=tf.float32, name='initial_state')

ForgetGateWeightHidden=tf.Variable(tf.random_normal([hidden_num,step_num]),dtype=tf.float32)
ForgetGateWeightInput=tf.Variable(tf.random_normal([elem_num,step_num]),dtype=tf.float32)
ForgetGateBias=tf.Variable(tf.random_normal([step_num]),dtype=tf.float32)

OutputGateWeightHidden=tf.Variable(tf.random_normal([hidden_num,step_num]),dtype=tf.float32)
OutputGateWeightInput=tf.Variable(tf.random_normal([elem_num,step_num]),dtype=tf.float32)
OutputGateBias=tf.Variable(tf.random_normal([step_num]),dtype=tf.float32)

outputHiddenStateWeight=tf.Variable(tf.random_normal([hidden_num,elem_num]),dtype=tf.float32)
outputHiddenStateBias=tf.Variable(tf.random_normal([elem_num]),dtype=tf.float32)


# Loop for refreshing the weights
def updateLoop(LSTMStates,inputX):
    CellState,HiddenState=tf.unstack(LSTMStates)
    Forget=tf.nn.sigmoid(tf.matmul(HiddenState,ForgetGateWeightHidden) + tf.matmul(inputX,ForgetGateWeightInput) + ForgetGateBias)
    CellState=tf.matmul(Forget,CellState)
    Output=tf.nn.sigmoid(tf.matmul(HiddenState,OutputGateWeightHidden) + tf.matmul(inputX,OutputGateWeightInput) + OutputGateBias)
    HiddenState=tf.matmul(Output,tf.tanh(CellState))
    return(tf.stack([CellState,HiddenState]))

# Running the loop
states = tf.scan(updateLoop,x,initializer=LSTMLoopInitState)

# Extract only the HiddenState
hiddenState=tf.transpose(states,[1,0,2,3])[0]
hiddenState=tf.reshape(hiddenState,(None,hidden_num))

# Get the Output                      
output=tf.nn.softmax(tf.matmul(hiddenState,outputHiddenStateWeight) + outputHiddenStateBias)

# Optimization
loss=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(output,outputY))
optimizer=tf.train.AdagradOptimizer(learning_rate=0.1).minimize(loss)

# Creating the input data
input_x=np.random.randint(100,5,1)
input_y=np.random.randint(100,5,1)

with tf.Session() as sess:
    for epoch in range(100):
        print(sess.run(loss, feed_dict={x: input_x, y: input_y}))

import tensorflow as tf
from tensorflow.python.ops.rnn_cell import LSTMCell

import numpy as np


class LSTMAutoencoder(object):

    """Basic version of LSTM-autoencoder.
  (cf. http://arxiv.org/abs/1502.04681)
  Usage:
    ae = LSTMAutoencoder(hidden_num, inputs)
    sess.run(ae.train)
  """

    def __init__(
        self,
        hidden_num,
        inputs,
        cell=None,
        optimizer=None,
        reverse=True,
        decode_without_input=False,
        ):
        """
    Args:
      hidden_num : number of hidden elements of each LSTM unit.
      inputs : a list of input tensors with size 
              (batch_num x elem_num)
      cell : an rnn cell object (the default option 
            is `tf.python.ops.rnn_cell.LSTMCell`)
      optimizer : optimizer for rnn (the default option is
              `tf.train.AdamOptimizer`)
      reverse : Option to decode in reverse order.
      decode_without_input : Option to decode without input.
    """
        self.batch_num = inputs[0].get_shape().as_list()[0]
        self.elem_num = inputs[0].get_shape().as_list()[1]
        if cell is None:
            self._enc_cell = LSTMCell(hidden_num)
            self._dec_cell = LSTMCell(hidden_num)
        else:
            self._enc_cell = cell
            self._dec_cell = cell
        with tf.variable_scope('encoder'):
            (self.z_codes, self.enc_state) = tf.contrib.rnn.static_rnn(self._enc_cell, inputs, dtype=tf.float32)

        with tf.variable_scope('decoder') as vs:
            dec_weight_ = tf.Variable(tf.truncated_normal([hidden_num,self.elem_num], dtype=tf.float32), name='dec_weight')
            dec_bias_ = tf.Variable(tf.constant(0.1,shape=[self.elem_num],dtype=tf.float32), name='dec_bias')
            if(1>0):
                dec_state = self.enc_state
                dec_input_ = tf.zeros(tf.shape(inputs[0]),dtype=tf.float32)
                dec_outputs = []
                for step in range(len(inputs)):
                    if step > 0:
                        vs.reuse_variables()
                    (dec_input_, dec_state) = self._dec_cell(dec_input_, dec_state)
                    dec_input_ = tf.matmul(dec_input_, dec_weight_) + dec_bias_
                    dec_outputs.append(dec_input_)
                if reverse:
                    dec_outputs = dec_outputs[::-1]
                self.output_ = tf.transpose(tf.stack(dec_outputs), [1,0,2])
                self.final_dec_weight=dec_weight_
                self.final_dec_bias=dec_bias_
        self.input_ = tf.transpose(tf.stack(inputs), [1, 0, 2])
        self.loss = tf.reduce_mean(tf.square(self.input_ - self.output_))
        self.enc_state_final=self.enc_state
        if optimizer is None:
            self.train = tf.train.AdamOptimizer().minimize(self.loss)
        else:
            self.train = optimizer.minimize(self.loss)

tf.reset_default_graph()
tf.set_random_seed(2016)
np.random.seed(2016)

# Constants
batch_num = 5
hidden_num = 12
step_num = 10
elem_num = 2
iteration = 4

# placeholder list
p_input = tf.placeholder(tf.float32, shape=(batch_num, step_num, elem_num))
p_inputs = [tf.squeeze(t, [1]) for t in tf.split(p_input, step_num, 1)]

cell = tf.nn.rnn_cell.LSTMCell(hidden_num, use_peepholes=True)
ae = LSTMAutoencoder(hidden_num, p_inputs, cell=cell, decode_without_input=False)

# Debugging Prints
print("Class Batch Size is {} and Class Elem Size is {}".format(ae.batch_num,ae.elem_num))

import pandas as pd
sess=tf.Session()
sess.run(tf.global_variables_initializer())
data=pd.DataFrame(np.random.randint(10,size=(50,2)),columns=['BITCOIN_RETURN','NVIDIA_RETURN'])
hiddenStateVar=[]
hiddenState=[]
encodedStateAppend=[]
for x in range(10):
    encoded_state,curLoss = sess.run([ae.enc_state_final,ae.loss], {p_input: data[['BITCOIN_RETURN','NVIDIA_RETURN']].values.reshape(5,10,2)})
    print(curLoss)
    encodedStateAppend.append(encoded_state)
    #print(weight)
    #hiddenStateVar.append(np.var(secondLastLayer[0][0]))
    #hiddenState.append(secondLastLayer[0][0])
print(encodedStateAppend)

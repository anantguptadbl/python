import tensorflow as tf
from tensorflow.python.ops.rnn_cell import LSTMCell

import numpy as np


class LSTMAutoencoder(object):
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
        # BATCH_NUM    : 
        # ELEM_NUM     :
        # HIDDEN_NUM   : 
        # STATE SIZE   :
        self.batch_num = inputs[0].get_shape().as_list()[0]
        self.elem_num = inputs[0].get_shape().as_list()[1]
        if cell is None:
            self._enc_cell = LSTMCell(hidden_num)
            self._dec_cell = LSTMCell(hidden_num)
        else:
            self._enc_cell = cell
            self._dec_cell = cell

        with tf.variable_scope('encoder'):
            # ENC_STATE  : SIZE is BATCH_SIZE, STATE_SIZE
            (self.z_codes, self.enc_state) = tf.contrib.rnn.static_rnn(self._enc_cell, inputs, dtype=tf.float32)
            # What is the shape of the enc_state??
            # the length o the z_codes will be equal to the step size

        with tf.variable_scope('decoder') as vs:
            dec_weight_ = tf.Variable(tf.truncated_normal([hidden_num,self.elem_num], dtype=tf.float32), name='dec_weight')
            dec_bias_ = tf.Variable(tf.constant(0.1,shape=[self.elem_num],dtype=tf.float32), name='dec_bias')
            # We can either have Decoder with input state
            if decode_without_input:
                dec_inputs = [tf.zeros(tf.shape(inputs[0]),dtype=tf.float32) for _ in range(len(inputs))]
                (dec_outputs, dec_state) = tf.contrib.rnn.static_rnn(self._dec_cell, dec_inputs, initial_state=self.enc_state,dtype=tf.float32)
                #if reverse:
                #    dec_outputs = dec_outputs[::-1]
                dec_output_ = tf.transpose(tf.stack(dec_outputs), [0,1,2])
                #dec_output_=dec_outputs
                dec_weight_ = tf.tile(tf.expand_dims(dec_weight_, 0),[len(inputs), 1, 1])
                self.output_ = tf.matmul(dec_output_, dec_weight_) + dec_bias_
            else:
            # This would not take the final encoded state as the input
                dec_state = self.enc_state
                dec_input_ = tf.zeros(tf.shape(inputs[0]),dtype=tf.float32)
                dec_outputs = []
                for step in range(len(inputs)):
                    if step > 0:
                        vs.reuse_variables()
                    (dec_input_, dec_state) = self._dec_cell(dec_input_, dec_state)
                    # This is done so that the output is converted from batch_size * hidden_num to batch_size * elem_num
                    dec_input_ = tf.matmul(dec_input_, dec_weight_) + dec_bias_
                    # The append is done so that for each of the range of step_size, we get the output
                    # The following will have a shape of step_size * batch_size * elem_num
                    dec_outputs.append(dec_input_)
                    
                #if reverse:
                #    dec_outputs = dec_outputs[::-1]
                # The following converts the step_size * batch_size * elem_num to batch_size * step_size * elem_num
                #self.output_ = tf.transpose(tf.stack(dec_outputs), [1,0,2])
                self.output_=dec_outputs
        # INPUTS : step_size * batch_num * elem_num
        # This is converted to : batch_num * step_size * elem_num
        # What is the reason for doing this?
        #self.input_ = tf.transpose(tf.stack(inputs), [1, 0, 2])
        self.input_=inputs
        self.loss = tf.reduce_mean(tf.square(self.input_ - self.output_))

        if optimizer is None:
            self.train = tf.train.AdamOptimizer().minimize(self.loss)
        else:
            self.train = optimizer.minimize(self.loss)
tf.reset_default_graph()
a=np.array([[1,2,3],[1,2,1],[1,1,1],[2,1,3],[2,2,2],[3,4,3],[3,3,3],[3,4,5],[5,6,7],[5,6,4],[5,4,3],[5,6,6]],dtype=np.int32)
a=a.reshape(4,3,3)
step_num=4
batch_num=3
elem_num=3
iteration=100
hidden_num=2
p_input = tf.placeholder(tf.float32, shape=(step_num, batch_num, elem_num))
p_inputs=[tf.squeeze(t,0) for t in tf.split(p_input,step_num,0)]

print("Input into the class is {}".format(a.shape))
print("Batch num calculated is {}".format(p_inputs[0].get_shape().as_list()[0]))
print("Elem num calculated is {}".format(p_inputs[0].get_shape().as_list()[1]))

ae = LSTMAutoencoder(hidden_num, p_inputs,decode_without_input=False)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(iteration):
        (loss_val, _) = sess.run([ae.loss, ae.train], {p_input:a})
        print('iter %d:' % (i + 1), loss_val)

    (input_, output_) = sess.run([ae.input_, ae.output_], {p_input:a})
    print("From the sess_run input_ = {} and output_ = {}".format(input_,output_))

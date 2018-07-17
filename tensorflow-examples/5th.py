import tensorflow as tf
import numpy as np

retData=np.array([0.01,-0.02,-0.01,-0.024,0.008,0.03,-0.021,-0.019,0.02,0.03]).reshape(1,10)
weights=np.array([0.2,0.1,0.2,0.15,0.35]).reshape(1,5)
covMatrix=np.random.rand(10,10)
# How to optimize the portfolio rejig

inp_returns=tf.placeholder("float32",(1,10))
inp_weights=tf.placeholder("float32",(1,5))
#var_weights=tf.Variable(tf.random_uniform([1,5]))
var_weights=tf.Variable(tf.zeros([1,5]))
inp_covMatrix=tf.placeholder("float32",(10,10))

x_weights=tf.concat([inp_weights,var_weights], 1)
#tf.get_variable(name='x_weights',initializer = tf.constant_initializer(weights),shape=(1,10))

# Factor 1 : Returns
ret=tf.matmul(inp_returns,tf.transpose(x_weights))
# Factor 2 : Deviation
devPortfolio=tf.matmul(tf.matmul(x_weights,inp_covMatrix),tf.transpose(x_weights))
# Factor 3: Number of changes in weights
changePortfolio=tf.equal(tf.concat([inp_weights,tf.zeros([1,5])], 1),x_weights)
changeWeight=tf.reduce_sum(tf.cast(changePortfolio, tf.float32))


# Init
init=tf.global_variables_initializer()

# Loss Function
loss=-tf.reduce_sum(ret) + tf.reduce_sum(devPortfolio) + changeWeight
optimizer=tf.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.Session() as sess:
    print("The session is {}".format(sess))
    sess.run(init)
    for x in range(10):
        lossVal=sess.run([loss],feed_dict={inp_returns:retData,inp_weights:weights,inp_covMatrix:covMatrix})
        print(lossVal)

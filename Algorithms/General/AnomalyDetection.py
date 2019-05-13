# Plain anomaly detection

import pandas as pd
import numpy as np
data=pd.DataFrame([10,2,3,4,5,0,2,3,0,4,0,4,0,4,0,0,4,0,4,0,0,4,0,4,0,0,4,0,4,0,0,4,0,4,0,0,4,0,4,0,0,4,0,4,0],columns=['data1'])

import tensorflow as tf
from keras import backend as K
tf.reset_default_graph()
import random
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers import Input, LSTM, RepeatVector, Convolution1D, MaxPooling1D, UpSampling1D
from tensorflow.python.keras.models import Model
from keras.backend.tensorflow_backend import set_session
np.random.seed(1337)
set_session(tf.Session())

n_batch = 9
n_epoch = 10000
timesteps = 5
input_dim = 1
latent_dim = 5

def make_lstm():
    inputs = Input(shape=(timesteps, input_dim))
    #lstm1, state_h, state_c = LSTM(latent_dim,return_state=True)(inputs)
    encoded , state_h, state_c = LSTM(latent_dim,return_state=True)(inputs)
    #encoded = LSTM(latent_dim)(inputs)
    decoded = RepeatVector(timesteps)(encoded)
    decoded = LSTM(input_dim, return_sequences=True)(decoded)
    model1 = Model(inputs=inputs, outputs=[encoded , state_h, state_c])
    sequence_autoencoder = Model(inputs, decoded)
    encoder = Model(inputs, encoded, decoded,state_h,state_c)

    sequence_autoencoder.compile(loss='mean_squared_error', optimizer='adam')
    return sequence_autoencoder, encoder, model1

sess=tf.Session()
sequence_autoencoder, encoder,model1 = make_lstm()
print(sequence_autoencoder.summary())

for epoch in range(n_epoch):
    x_train=data['data1'].values.reshape(n_batch,timesteps,1)
    sequence_autoencoder.fit(x_train, x_train, epochs=1, steps_per_epoch=1,verbose=0)

hiddenWeights=sequence_autoencoder.layers[1].get_weights()[1]
print("The LSTM AutoEncoder has been trained")
print("Shape of hidden weights are {}".format(hiddenWeights.shape))

#print(sum(sum((a-hiddenWeights)**2)))

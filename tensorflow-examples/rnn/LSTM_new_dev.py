import tensorflow as tf
tf.reset_default_graph()

class LSTM_rnn():

    def __init__(self, batch_size,state_size, num_classes,hidden_state):
        self.batch_size=batch_size
        self.state_size = state_size
        self.num_classes = num_classes
        self.hidden_state=hidden_state
        #self.ckpt_path = ckpt_path
        #self.model_name = model_name

        # build graph ops
        def __graph__():
            tf.reset_default_graph()
            # inputs
            xs_ = tf.placeholder(shape=[self.batch_size,self.state_size, num_classes], dtype=tf.float32)
            ys_ = tf.placeholder(shape=[1], dtype=tf.float32)
            #
            # embeddings
            #embs = tf.get_variable('emb', [num_classes, state_size])
            #rnn_inputs = tf.nn.embedding_lookup(embs, xs_)
            rnn_inputs = xs_
            #
            # Initial hidden state
            #init_state = tf.placeholder(shape=[2, self.batch_size,self.hidden_elem], dtype=tf.float32, name='initial_state')
            #init_cell_state=tf.placeholder(shape=[self.batch_size,self.batch_size],dtype=tf.float32,name='cellState')
            #init_hidden_state=tf.placeholder(shape=[self.batch_size,self.hidden_state],dtype=tf.float32,name='hiddenState')
            LSTMLoopInitState=tf.placeholder(shape=[2,None,step_num], dtype=tf.float32, name='initial_state')
            # Initializer
            xav_init = tf.contrib.layers.xavier_initializer
            # Params
            W = tf.get_variable('W', shape=[4,self.state_size,self.num_classes], initializer=xav_init())
            #U = tf.get_variable('U', shape=[3, self.state_size, self.hidden_elem], initializer=xav_init())
            UInputGate=tf.get_variable('UInputGate', shape=[self.num_classes,self.batch_size], initializer=xav_init())
            UForgetGate=tf.get_variable('UForgetGate', shape=[self.num_classes,self.batch_size], initializer=xav_init())
            UOutputGate=tf.get_variable('UOutputGate', shape=[self.num_classes,self.batch_size], initializer=xav_init())
            UGate=tf.get_variable('UGate', shape=[self.num_classes,self.batch_size], initializer=xav_init())
            #b = tf.get_variable('b', shape=[self.state_size], initializer=tf.constant_initializer(0.))
            ####
            # step - LSTM
            def step(prev, x):
                # gather previous internal state and output state
                print("Type of the variable prev is {}".format(prev))
                st_1, ct_1 = tf.unstack(prev)
                ####
                # GATES
                #  input gate
                i = tf.sigmoid(tf.matmul(x,UInputGate) + tf.matmul(st_1,W[0]))
                #  forget gate
                f = tf.sigmoid(tf.matmul(x,UForgetGate) + tf.matmul(st_1,W[1]))
                #  output gate
                o = tf.sigmoid(tf.matmul(x,UOutputGate) + tf.matmul(st_1,W[2]))
                #  gate weights
                g = tf.tanh(tf.matmul(x,UGate) + tf.matmul(st_1,W[3]))
                ###
                # new internal cell state
                ct = f*ct_1 + i*g
                
                # output state
                st = o*tf.tanh(ct)
                return tf.pack([st, ct])
            ###
            # here comes the scan operation; wake up!
            #   tf.scan(fn, elems, initializer)
            states = tf.scan(step, 
                    tf.transpose(rnn_inputs, [1,0,2]),
                    initializer=LSTMLoopInitState)
                    #initializer=[init_cell_state,init_hidden_state])
            #
            # predictions
            V = tf.get_variable('V', shape=[state_size, num_classes], 
                                initializer=xav_init())
            bo = tf.get_variable('bo', shape=[num_classes], 
                                 initializer=tf.constant_initializer(0.))

            ####
            # get last state before reshape/transpose
            last_state = states[-1]

            ####
            # transpose
            states = tf.transpose(states, [1,2,0,3])[0]
            #st_shp = tf.shape(states)
            # flatten states to 2d matrix for matmult with V
            #states_reshaped = tf.reshape(states, [st_shp[0] * st_shp[1], st_shp[2]])
            states_reshaped = tf.reshape(states, [-1, state_size])
            logits = tf.matmul(states_reshaped, V) + bo
            # predictions
            predictions = tf.nn.softmax(logits) 
            #
            # optimization
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, ys_)
            loss = tf.reduce_mean(losses)
            train_op = tf.train.AdagradOptimizer(learning_rate=0.1).minimize(loss)
            #
            # expose symbols
            self.xs_ = xs_
            self.ys_ = ys_
            self.loss = loss
            self.train_op = train_op
            self.predictions = predictions
            self.last_state = last_state
            self.init_state = init_state
        ##### 
        # build graph
        #sys.stdout.write('\n<log> Building Graph...')
        __graph__()
        #sys.stdout.write('</log>\n')

    ####
    # training
    def train(self, epochs=100):
        # training session
        inpData=np.random.rand([10,10,2],dtype=np.float32)
        outData=np.random.rand([10,1],dtype=np.float32)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            train_loss = 0
            try:
                for i in range(epochs):
                    for j in range(100):
                        xs, ys = train_set.__next__()
                        batch_size = xs.shape[0]
                        _, train_loss_ = sess.run([self.train_op, self.loss], feed_dict = {
                                self.xs_ : inpData,
                                self.ys_ : ys.flatten(),
                                self.init_state : np.zeros([2, batch_size, self.state_size])
                            })
                        train_loss += train_loss_
                    print('[{}] loss : {}'.format(i,train_loss/100))
                    train_loss = 0
            except KeyboardInterrupt:
                print('interrupted by user at ' + str(i))
                
l1=LSTM_rnn(10,10,2,2)
l1.train()

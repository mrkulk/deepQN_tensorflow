import tensorflow as tf

class Model(object):
    def __init__(self, params, src_net):
        self.X = tf.placeholder("float", [None, 210, 160, 3]) #input image
        self.actions = tf.placeholder("float", [None, params['num_actions']])
        self.rewards = tf.placeholder("float", [None, 1])
        self.terminals = tf.placeholder("float", [None, 1])
        
        if src_net is not None: #copy network
          self.w = tf.Variable(src_net.w); self.w2 = tf.Variable(src_net.w2);
          self.w3 = tf.Variable(src_net.w3); self.w4 = tf.Variable(src_net.w4)
          self.w_o = tf.Variable(src_net.w_o)
        else: #initialize network from scratch
          self.w = self.init_weights([8, 8, 3, 32] )
          self.w2 = self.init_weights([4, 4, 32, 64])
          self.w3 = self.init_weights([3, 3, 64, 64])
          self.w4 = self.init_weights([64 * 4 * 3, 512])
          self.w_o = self.init_weights([512, params['num_actions']])

        self.param_list = [self.w, self.w2, self.w3, self.w4, self.w_o]
        
        self.l1a = tf.nn.relu(tf.nn.conv2d(self.X, self.w, [1, 4, 4, 1], 'SAME'))
        self.l1 = tf.nn.max_pool(self.l1a, ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1], padding='SAME')

        self.l2a = tf.nn.relu(tf.nn.conv2d(self.l1, self.w2, [1, 2, 2, 1], 'SAME'))
        self.l2 = tf.nn.max_pool(self.l2a, ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1], padding='SAME')

        self.l3a = tf.nn.relu(tf.nn.conv2d(self.l2, self.w3, [1, 1, 1, 1], 'SAME'))
        self.l3 = tf.nn.max_pool(self.l3a, ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1], padding='SAME')
        self.l3 = tf.reshape(self.l3, [-1, self.w4.get_shape().as_list()[0]])

        self.l4 = tf.nn.relu(tf.matmul(self.l3, self.w4))

        self.pyx = tf.matmul(self.l4, self.w_o)

    def init_weights(self, shape):
        return tf.Variable(tf.random_normal(shape, stddev=0.01))




class curiosity(object):
    def __init__(self, params, src_net):
        self.X = tf.placeholder("float", [None, 210, 160, 3]) #input image
        self.actions = tf.placeholder("float", [None, params['num_actions']])
        self.rewards = tf.placeholder("float", [None, 1])
        self.terminals = tf.placeholder("float", [None, 1])

        obj_vec_size = params['max_num_objects'] * (2 + params['max_object_templates']) #2 coordinates + the size of the one hot vector
        self.objects = tf.placeholder("float", [None, obj_vec_size])

        if src_net is not None: #copy network
          self.w_cnn = tf.Variable(src_net.w_cnn); self.w2_cnn = tf.Variable(src_net.w2_cnn);
          self.w3_cnn = tf.Variable(src_net.w3_cnn); self.w4_cnn = tf.Variable(src_net.w4_cnn)
          self.w_o_cnn = tf.Variable(src_net.w_o_cnn)

          self.w_mlp1 = tf.Variable(src_net.w_mlp1)
          self.w2_mlp1 = tf.Variable(src_net.w2_mlp1)
          self.w_o_mlp1 = tf.Variable(src_net.w_o_mlp1)

        else: #initialize network from scratch
          self.w_cnn = self.init_weights([8, 8, 3, 32] )
          self.w2_cnn = self.init_weights([4, 4, 32, 64])
          self.w3_cnn = self.init_weights([3, 3, 64, 64])
          self.w4_cnn = self.init_weights([64 * 4 * 3, 512])

          self.w_mlp1 = self.init_weights([obj_vec_size + 512, 300])
          self.w2_mlp1 = self.init_weights([300, 300])
          self.w_o_mlp1 = self.init_weights([300, params['max_num_objects'] + params['num_actions']])

          # self.w_o = self.init_weights([512, params['num_actions']])

        self.param_list = [self.w_cnn, self.w2_cnn, self.w3_cnn, self.w4_cnn,
                           self.w_mlp1, self.w2_mlp1, self.w_o_mlp1]

        self.l1a = tf.nn.relu(tf.nn.conv2d(self.X, self.w, [1, 4, 4, 1], 'SAME'))
        self.l1 = tf.nn.max_pool(self.l1a, ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1], padding='SAME')

        self.l2a = tf.nn.relu(tf.nn.conv2d(self.l1, self.w2, [1, 2, 2, 1], 'SAME'))
        self.l2 = tf.nn.max_pool(self.l2a, ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1], padding='SAME')

        self.l3a = tf.nn.relu(tf.nn.conv2d(self.l2, self.w3, [1, 1, 1, 1], 'SAME'))
        self.l3 = tf.nn.max_pool(self.l3a, ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1], padding='SAME')
        self.l3 = tf.reshape(self.l3, [-1, self.w4.get_shape().as_list()[0]])

        self.l4 = tf.nn.relu(tf.matmul(self.l3, self.w4))

        self.mlp1_input = tf.np.vstack((self.objects, self.l4))
        self.l1_mlp1 = tf.nn.relu(tf.matmul(self.mlp1_input, self.w_mlp1))
        self.l2_mlp1 = tf.nn.relu(tf.matmul(self.l1_mlp1, self.w2_mlp1))
        self.l3_mlp1 = tf.nn.relu(tf.matmul(self.l2_mlp1, self.w_o_mlp1))

        self.pyx = tf.matmul(self.l3_mlp1, self.w_o_mlp1)

    def init_weights(self, shape):
        return tf.Variable(tf.random_normal(shape, stddev=0.01))
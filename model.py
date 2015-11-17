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
          self.w = self.init_weights([3, 3, 3, 32] )
          self.w2 = self.init_weights([3, 3, 32, 64])
          self.w3 = self.init_weights([3, 3, 64, 64])
          self.w4 = self.init_weights([64 * 4 * 4, 512])
          self.w_o = self.init_weights([512, params['num_actions']])

        self.param_list = [self.w, self.w2, self.w3, self.w4, self.w_o]
        
        self.l1a = tf.nn.relu(tf.nn.conv2d(self.X, self.w, [1, 1, 1, 1], 'SAME'))
        self.l1 = tf.nn.max_pool(self.l1a, ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1], padding='SAME')

        self.l2a = tf.nn.relu(tf.nn.conv2d(self.l1, self.w2, [1, 1, 1, 1], 'SAME'))
        self.l2 = tf.nn.max_pool(self.l2a, ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1], padding='SAME')

        self.l3a = tf.nn.relu(tf.nn.conv2d(self.l2, self.w3, [1, 1, 1, 1], 'SAME'))
        self.l3 = tf.nn.max_pool(self.l3a, ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1], padding='SAME')
        self.l3 = tf.reshape(self.l3, [-1, self.w4.get_shape().as_list()[0]])

        self.l4 = tf.nn.relu(tf.matmul(self.l3, self.w4))

        self.pyx = tf.matmul(self.l4, self.w_o)

        # self.cost = tf.add(self.reward, tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.pyx, self.Y)))

    def init_weights(self, shape):
        return tf.Variable(tf.random_normal(shape, stddev=0.01))

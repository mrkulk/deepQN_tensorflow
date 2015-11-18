import tensorflow as tf
import numpy as np
import input_data
from model import *
from emulator import *
from database import *
import copy

params = {
  'epochs': 10,
  'db_size': 1000,
  'bsize': 100,
  'num_actions': 10,
  'iter_qnet_to_target_copy': 1000,
  'input_dims' : [210, 160, 3]
}

DB = database(params['db_size'], params['input_dims'])
engine = emulator('montezuma_revenge.bin')

#creating Q and target network. 
qnet = Model(params, None)

#TODO
def select_action(state):
  return engine.possible_actions[0]

def perceive(newstate):
  if not newstate.terminal: 
    action = select_action(newstate)
    return action

def get_cost(nextstate):
  maxval = tf.reduce_max(targetnet.pyx, 1)
  y_j = tf.add(qnet.rewards, tf.mul(qnet.terminals, maxval))
  #we do not want to backprop and only need the value from the target network
  yj_val = sess.run(y_j, feed_dict = {targetnet.X: teX[0:params['bsize']], targetnet.Y: teY[0:params['bsize']]}) #TODO check
  Q_pred = tf.reduce_sum(qnet.pyx*qnet.actions, reduction_indices=[1,])
  yj_val = tf.Variable(yj_val); sess.run(tf.initialize_variables([yj_val]))
  return tf.pow(tf.sub(yj_val, Q_pred), 2)

def update_params():
  # randomly sample a mini-batch
  indxs = np.random.permutation(DB.size)[:params['bsize']]
  states, actions, terminals, nextstates, rewards = DB.get_batches(indxs)
  sess.run(train_op, feed_dict={qnet.X: states, targetnet.X: nextstates, qnet.actions: actions, qnet.terminals:terminals, qnet.rewards: rewards})

def train():
  for e in range(params['epochs']):
    for numeps in range(params['num_episodes']):
      prevstate = None; action = None
      for maxl in range(params['episode_max_length']):
        newstate, reward = engine.next(action) #IMP: newstate contains terminal info

        #store transition
        if prevstate:
          DB.insert({'s': prevstate, 's_': newstate, 'r':reward, 'a':action, 't' : newstate.terminal})

        action = perceive(newstate)
        update_params()


cost = get_cost()
train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)



# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
# trX = trX.reshape(-1, 28, 28, 1)
# teX = teX.reshape(-1, 28, 28, 1)

# sess = tf.Session()
# init = tf.initialize_all_variables()
# sess.run(init)
# targetnet = Model(params, qnet)
# sess.run(tf.initialize_variables(targetnet.param_list))



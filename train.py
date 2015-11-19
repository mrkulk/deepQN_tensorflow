import tensorflow as tf
import numpy as np
import input_data
from model import *
from emulator import *
from database import *
import copy
import pdb

params = {
  'epochs': 10,
  'db_size': 1000,
  'bsize': 100,
  'num_actions': 10,
  'iter_qnet_to_target_copy': 1000,
  'input_dims' : [210, 160, 3],
  'num_episodes': 100,
  'episode_max_length': 1000,
  'update_delay': 1000
}

DB = database(params['db_size'], params['input_dims'])
engine = emulator(rom_name='montezuma_revenge.bin', vis=True)

#creating Q and target network. 
qnet = Model(params, None)
#TODO
def select_action(state):
  return 0

def perceive(newstate, terminal):
  if not terminal: 
    action = select_action(newstate)
    return action

def get_cost(nextstates, actions, terminals, rewards):
  maxval = tf.reduce_max(targetnet.pyx, 1)
  y_j = tf.add(targetnet.rewards, tf.mul(targetnet.terminals, maxval))
  #we do not want to backprop and only need the value from the target network
  # yj_val = sess.run(y_j, feed_dict = {targetnet.X: nextstates, targetnet.actions: np.zeros((params['bsize'],params['num_actions'])), targetnet.terminals:np.zeros((params['bsize'],1)), targetnet.rewards: np.zeros((params['bsize'],1))}) #TODO check
  yj_val = sess.run(y_j, feed_dict = {targetnet.X: nextstates, targetnet.actions: actions, targetnet.terminals: terminals, targetnet.rewards: rewards }) #TODO check  
  yj_val = tf.Variable(yj_val); sess.run(tf.initialize_variables([yj_val]))

  Q_pred = tf.reduce_sum(qnet.pyx*qnet.actions, reduction_indices=[1,])
  loss = tf.pow(tf.sub(yj_val, Q_pred), 2)
  # sess.run(tf.initialize_variables([Q_pred, loss]))
  return loss

def get_onehot(actions):
  actions_onehot = np.zeros((params['bsize'], params['num_actions']))
  for i in range(len(actions)):
    actions_onehot[i][actions[i]] = 1
  return actions_onehot

def update_params():
  # randomly sample a mini-batch
  indxs = np.random.permutation(DB.size)[:params['bsize']]
  states, actions, terminals, nextstates, rewards = DB.get_batches(indxs)
  actions = get_onehot(actions)
  # print(np.shape(actions), np.shape(terminals), np.shape(rewards))
  cost = get_cost(nextstates, actions, terminals, rewards)
  # sess.run(tf.initialize_variables([cost]))

  train_op = tf.train.GradientDescentOptimizer(0.00001).minimize(cost); 
  # sess.run(tf.initialize_all_variables())
  sess.run(train_op, feed_dict={qnet.X: states, qnet.actions: actions, qnet.terminals:terminals, qnet.rewards: rewards})

def train():
  cnt = 1
  for e in range(params['epochs']):
    for numeps in range(params['num_episodes']):
      prevstate = None; action = None; terminal = None
      newstate = engine.newGame()
      print '\n'
      for maxl in range(params['episode_max_length']):
        if prevstate is not None:
          DB.insert({'s': prevstate, 's_': newstate, 'r':reward, 'a':action, 't' : terminal})
          cnt = cnt + 1
        action = perceive(newstate, terminal)
        if cnt >= params['update_delay']:
          print '.' ,
          update_params()
        prevstate = newstate
        newstate, reward, terminal = engine.next(action) #IMP: newstate contains terminal info

def unit_test():
  qnet = Model(params, None)
  sess = tf.Session()
  init = tf.initialize_all_variables()
  sess.run(init)
 
  targetnet = Model(params, qnet)
  sess.run(tf.initialize_variables(targetnet.param_list))

  maxval = tf.reduce_max(targetnet.pyx, 1)
  y_j = tf.add(targetnet.rewards, tf.mul(targetnet.terminals, maxval))
  yj_val = sess.run(y_j, feed_dict = {targetnet.X: np.zeros((params['bsize'],210, 160,3)), targetnet.actions: np.zeros((params['bsize'],params['num_actions'])), targetnet.terminals:np.zeros((params['bsize'],1)), targetnet.rewards: np.zeros((params['bsize'],1))}) #TODO check
  print(':', np.shape(yj_val))

# unit_test()

# creating Q and target network. 
qnet = Model(params, None)
sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)
targetnet = Model(params, qnet)
sess.run(tf.initialize_variables(targetnet.param_list))
train()

# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
# trX = trX.reshape(-1, 28, 28, 1)
# teX = teX.reshape(-1, 28, 28, 1)

# print(np.shape(trX))
# sess = tf.Session()
# init = tf.initialize_all_variables()
# sess.run(init)
# targetnet = Model(params, qnet)
# sess.run(tf.initialize_variables(targetnet.param_list))



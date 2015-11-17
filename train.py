import tensorflow as tf
import numpy as np
import input_data
from model import *
from emulator import *
from database import *
import copy

params = {
  'epochs': 10,
  'dictsize': 1000,
  'bsize': 100,
  'num_actions': 10
}
DB = database()
engine = emulator('montezuma_revenge.bin')

def unit_test_shared():
  mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
  trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
  trX = trX.reshape(-1, 28, 28, 1)
  teX = teX.reshape(-1, 28, 28, 1)

  qnet = Model(params, None)
  targetnet = Model(params, None)

  sess = tf.Session()
  init = tf.initialize_all_variables()
  sess.run(init)

  print(sess.run(qnet.pyx, feed_dict = {qnet.X: teX[0:128], qnet.Y: teY[0:128], qnet.reward:np.zeros(1)+100}))
  print('---\n')
  print(sess.run(targetnet.pyx, feed_dict = {targetnet.X: teX[0:128], targetnet.Y: teY[0:128], targetnet.reward:np.zeros(1)+100}))
  print('----\n')
  targetnet = Model(params, qnet)
  sess.run(tf.initialize_variables(targetnet.param_list))
  print(sess.run(targetnet.pyx, feed_dict = {targetnet.X: teX[0:128], targetnet.Y: teY[0:128], targetnet.reward:np.zeros(1)+100}))
  print('----\n')
  sess.run(tf.initialize_variables(qnet.param_list))
  print(sess.run(qnet.pyx, feed_dict = {qnet.X: teX[0:128], qnet.Y: teY[0:128], qnet.reward:np.zeros(1)+100}))
  print('----\n')
  print(sess.run(targetnet.pyx, feed_dict = {targetnet.X: teX[0:128], targetnet.Y: teY[0:128], targetnet.reward:np.zeros(1)+100}))


#creating Q and target network. 
qnet = Model(params, None)
targetnet = Model(params, qnet)

def select_action(state):
  return engine.possible_actions[0]

def perceive(prevstate, newstate, reward):
  DB.insert({'s': prevstate, 's_': newstate, 'reward':reward})
  if not newstate.terminal: 
    action = select_action(newstate)
    return action

def get_cost(states, actions, terminals, nextstates, rewards):
  maxval = tf.reduce_max(targetnet.pyx, 1) #TODO - check and add discount factor
  y_j = tf.add(rewards, tf.mul(terminals, maxval))
  yj_val = sess.run(y_j, feed_dict = {targetnet.X: teX[0:128], targetnet.Y: teY[0:128], targetnet.reward:np.zeros(1)+100})
  cost = tf.gather(qnet.pyx, actions)
  return tf.add(rewards, tf.do)

def update_params():
  # randomly sample a mini-batch
  indxs = np.random.permutation(DB.size())[:params.bsize]
  states, actions, terminals, nextstates, rewards = DB.get_batches(indxs)
  cost = get_cost(states, actions, terminals, nextstates, rewards)


def train():
  for e in range(params.epochs):
    for numeps in range(params.num_episodes):
      prevstate = None; action = None
      for maxl in range(params.episode_max_length):
        newstate, reward = engine.next(action)
        action = perceive(prevstate, newstate, reward)
        update_params()
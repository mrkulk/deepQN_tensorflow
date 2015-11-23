import tensorflow as tf
import numpy as np
from model import *
from emulator import *
from database import *
import copy
import pdb
#import matplotlib.pyplot as plt
#from matplotlib.backends.backend_pdf import PdfPages
import sys
import gc, time

gc.enable()

params = {
  'epochs': 10000,
  'db_size': 100000,
  'bsize': 32,
  'num_actions': 10,
  'iter_qnet_to_target_copy': 1000,
  'input_dims' : [210, 160, 3],
  'num_episodes': 100,
  'episode_max_length': 10000,
  'update_params': 4,
  'copy_qnet': 1000,
  'eps': 1,
  'discount': 0.99,
  'lr': 0.0005,
  'plot_delay': 1000
}

def select_action(state):
  if np.random.rand() > params['eps']:
    #greedy with random tie-breaking
    Q_pred = sess.run(qnet.pyx, feed_dict = {qnet.X: np.reshape(state, (1,210,160,3)), qnet.actions: np.zeros((params['bsize'],params['num_actions'])), qnet.terminals:np.zeros((params['bsize'],1)), qnet.rewards: np.zeros((params['bsize'],1))})[0] #TODO check
    a_winner = np.argwhere(Q_pred == np.amax(Q_pred))
    if len(a_winner) > 1:
      return engine.legal_actions[a_winner[np.random.randint(0, len(a_winner))][0]]
    else:
      return engine.legal_actions[a_winner[0][0]]
  else:
    #random
    return engine.legal_actions[np.random.randint(0,len(engine.legal_actions))]

def perceive(newstate, terminal):
  if not terminal: 
    action = select_action(newstate)
    return action

def get_onehot(actions):
  actions_onehot = np.zeros((params['bsize'], params['num_actions']))
  for i in range(len(actions)):
    actions_onehot[i][engine.action_map[int(actions[i])]] = 1
  return actions_onehot

def update_params():
  global tflow_opt
  # randomly sample a mini-batch
  indxs = np.random.permutation(DB.get_size())[:params['bsize']]
  states, actions, terminals, nextstates, rewards = DB.get_batches(indxs)
  actions = get_onehot(actions)
  sess.run(tflow_opt, feed_dict={qnet.X: states, qnet.actions: actions, qnet.terminals:terminals, qnet.rewards: rewards, targetnet.X: nextstates, targetnet.actions: actions, targetnet.terminals: terminals, targetnet.rewards: rewards })

def train():
  global global_cntr
  for e in range(params['epochs']):
    for numeps in range(params['num_episodes']):
      prevstate = None; action = None; terminal = None
      newstate = engine.newGame()
      cnt = 1; delay = 1;start_time = time.time()
      total_reward_ep = 0
      for maxl in range(params['episode_max_length']):
        start = time.time()
        if prevstate is not None:
          DB.insert({'s': prevstate, 's_': newstate, 'r':reward, 'a':action, 't' : terminal})
          cnt = cnt + 1
        action = perceive(newstate, terminal)
        if action == None: #TODO - check [terminal condition]
          break
        if cnt % params['update_params'] == 0 and global_cntr > 1000:
          update_params()
          delay = delay + 1
     
        if delay % params['copy_qnet'] == 0:
          #copy qnet to targetnet
          print 'Copying qnet to targetnet'
          targetnet = Model(params, qnet)
          sess.run(tf.initialize_variables(targetnet.param_list))
          
        prevstate = newstate
        newstate, reward, terminal = engine.next(action) #IMP: newstate contains terminal info
        params['eps'] = 0.1 + max(0, (1 - 0.1) * (100000 - max(0, global_cntr))/100000)
        total_reward_ep = total_reward_ep + reward
        global_cntr = global_cntr + 1

      sys.stdout.write("Episode: %d | Training progress: %d | ep_time: %f | reward: %f \r" % (numeps, global_cntr, time.time()-start_time, total_reward_ep))
      sys.stdout.flush()


if __name__ == "__main__":
  qvals = []
  global_cntr = 1
  DB = database(params['db_size'], params['input_dims'])
  engine = emulator(rom_name='breakout.bin', vis=False)
  params['num_actions'] = len(engine.legal_actions)

  # creating Q and target network. 
  qnet = Model(params, None)
  sess = tf.Session()
  init = tf.initialize_all_variables()
  sess.run(init)
  targetnet = Model(params, qnet)
  sess.run(tf.initialize_variables(targetnet.param_list))

  #cost calculation
  discount = tf.constant(params['discount'])
  maxval = tf.mul(discount, tf.reduce_max(targetnet.pyx, 1))
  yj_val = tf.add(targetnet.rewards, tf.mul(targetnet.terminals, maxval))

  Q_pred = tf.reduce_sum(qnet.pyx*qnet.actions, reduction_indices=[1,])
  cost = tf.pow(tf.sub(yj_val, Q_pred), 2)

  train_op = tf.train.GradientDescentOptimizer(params['lr'])
  grads_and_vars = train_op.compute_gradients(cost, qnet.param_list)
  clipped_grads_and_vars = [(tf.clip_by_value(gv[0],-10.0,10.0), gv[1]) for gv in grads_and_vars]
  tflow_opt = train_op.apply_gradients(clipped_grads_and_vars)

  train()

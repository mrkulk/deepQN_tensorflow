import numpy as np

class database:
  def __init__(self, size, input_dims):
    #create database with input_dims as list of input dimensions
    self.size = size
    self.states = np.zeros([self.size] + input_dims) #image dimensions
    self.actions = np.zeros(self.size)
    self.terminals = np.zeros((self.size,1))
    self.nextstates = np.zeros([self.size] + input_dims)
    self.rewards = np.zeros((self.size,1))

    self.counter = 0 #keep track of next empty state

    return

  def get_batches(self, indxs):
    return self.states[indxs], self.actions[indxs], self.terminals[indxs], self.nextstates[indxs], self.rewards[indxs]

  def insert(self, dic):
    self.states[self.counter] = dic['s']
    self.nextstates[self.counter] = dic['s_']
    self.rewards[self.counter][0] = dic['r']
    self.actions[self.counter] = dic['a']
    self.terminals[self.counter][0] = dic['t']

    #update counter
    self.counter += 1
    if self.counter >= self.size:
      self.counter = 0

    return

#all unit tests

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

  print(sess.run(qnet.pyx, feed_dict = {qnet.X: teX[0:params['bsize']], qnet.Y: teY[0:params['bsize']], qnet.reward:np.zeros(1)+100}))
  print('---\n')
  print(sess.run(targetnet.pyx, feed_dict = {targetnet.X: teX[0:params['bsize']], targetnet.Y: teY[0:params['bsize']], targetnet.reward:np.zeros(1)+100}))
  print('----\n')
  targetnet = Model(params, qnet)
  sess.run(tf.initialize_variables(targetnet.param_list))
  print(sess.run(targetnet.pyx, feed_dict = {targetnet.X: teX[0:params['bsize']], targetnet.Y: teY[0:params['bsize']], targetnet.reward:np.zeros(1)+100}))
  print('----\n')
  sess.run(tf.initialize_variables(qnet.param_list))
  print(sess.run(qnet.pyx, feed_dict = {qnet.X: teX[0:params['bsize']], qnet.Y: teY[0:params['bsize']], qnet.reward:np.zeros(1)+100}))
  print('----\n')
  print(sess.run(targetnet.pyx, feed_dict = {targetnet.X: teX[0:params['bsize']], targetnet.Y: teY[0:params['bsize']], targetnet.reward:np.zeros(1)+100}))


def unit_test_cost():
  actions = np.random.randint(0, 10, (params['bsize'],))
  actions_onehot = np.zeros((params['bsize'], params['num_actions']))
  for i in range(len(actions)):
    actions_onehot[i,actions[i]] = 1

  actions_onehot = np.float32(actions_onehot)
  actions_onehot = tf.Variable(actions_onehot)
  terminals = tf.Variable(np.float32(np.random.randint(0, 2, (params['bsize'],))))
  rewards = tf.Variable(np.float32(10.0*np.random.randint(0, 2, (params['bsize'],))))
  sess.run(tf.initialize_variables([actions_onehot, terminals, rewards]))
  out = get_cost(None, actions_onehot, terminals,None, rewards)
  res = sess.run(out, feed_dict = {
    targetnet.X: teX[0:params['bsize']], targetnet.Y: teY[0:params['bsize']], #current state
    qnet.X: teX[0:params['bsize']], qnet.Y: teY[0:params['bsize']], #next state
  })
  print(np.shape(res))
  print(res[0:10])
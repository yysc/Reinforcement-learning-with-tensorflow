import tensorflow as tf
import pandas as pd
import numpy as np

np.random.seed(1)
tf.set_random_seed(1)


class DeepQNetwork:
    def __init__(
        self,
        n_actions,
        n_features,
        learning_rate=0.01,
        reward_decay=0.9,
        e_greedy=0.9,
        replace_target_iter=300,
        memory_size=500,
        batch_size=32,
        e_greedy_increment=None,
        output_graph=False
    ):
        self.n_actions = n_actions
        # state observation.
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        # update target frequency.
        self.replace_target_iter = replace_target_iter
        # experience mount
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        self.learn_step_counter = 0
        # observation this round + observation next round + reward + action this round
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        self._build_net()

        self.sess = tf.Session()

        if output_graph:
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []

    def _build_net(self):
        # state from enviroment.
        self.state = tf.placeholder(
            tf.float32, [None, self.n_features], name='state_this_round')
        # target value from target_net
        self.q_target = tf.placeholder(
            tf.float32, [None, self.n_actions], name='Q_target')
        # use eval net to predict q.
        with tf.variable_scope('eval_net'):
            c_names, n_l1, w_initializer, b_initializer = [
                'eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 10, tf.random_normal_initializer(0, 0.3), tf.constant_initializer(0.1)
            # eval first layer.
            with tf.variable_scope('l1'):
                w1 = tf.get_variable(
                    'w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable(
                    'b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.state, w1) + b1)
            with tf.variable_scope('l2'):
                w2 = tf.get_variable(
                    'w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable(
                    'b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_eval_net = tf.matmul(l1, w2) + b2
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(
                tf.squared_difference(self.q_target, self.q_eval_net))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(
                self.lr).minimize(self.loss)

        self.s_ = tf.placeholder(
            tf.float32, [None, self.n_features], name='state_next_round')
        with tf.variable_scope('target_net'):
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            with tf.variable_scope('l1'):
                w1 = tf.get_variable(
                    'w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable(
                    'b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1)

            with tf.variable_scope('l2'):
                w2 = tf.get_variable(
                    'w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable(
                    'b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_next_net = tf.matmul(l1, w2) + b2

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, observation):
        observation = observation[np.newaxis, :]

        if np.random.uniform() < self.epsilon:
            actions_value = self.sess.run(self.q_eval_net, feed_dict={
                                          self.state: observation})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def _replace_target_params(self):
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.sess.run([tf.assign(t, e) for t, e in zip(t_params, e_params)])

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self._replace_target_params()
            print('\n update target parameter.')

        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(
                self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(
                self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        # sample memory and assign to variable.
        # q_next :[1,n_actions] q_eval :[1,n_actions]
        q_next, q_eval = self.sess.run([self.q_next_net, self.q_eval_net], feed_dict={
            self.s_: batch_memory[:, -self.n_features:],
            self.state: batch_memory[:, :self.n_features]
        })

        # change target to eval actions.
        q_target = q_eval.copy()
        # batch index array[0,1,2,3,...]
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        # action value
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        # reward value
        reward = batch_memory[:, self.n_features + 1]

        # only update current q(s,a) and q(s_,a_)
        q_target[batch_index, eval_act_index] = reward + \
            self.gamma * np.max(q_next, axis=1)
        _, self.cost = self.sess.run([self._train_op, self.loss], feed_dict={
                                     self.state: batch_memory[:, :self.n_features], self.q_target: q_target})
        self.cost_his.append(self.cost)

        self.epsilon = self.epsilon + \
            self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('Training steps')
        plt.show()

import numpy as np
import tensorflow as tf

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
        out_graph=False
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        self.learn_step_counter = 0
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        self._build_net()
        self.sess = tf.Session()
        if out_graph:
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []

    def _build_net(self):
        def build_layers(s, c_names, n_l1, w_initializer, b_initializer):
            with tf.variable_scope('l1'):
                # [1,n_features]*[n_features,n_l1]
                w1 = tf.get_variable(
                    'w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable(
                    'b1', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                l1 = tf.relu(tf.matmul(s, w1) + b1)
            with tf.variable_scope('l2'):
                w2 = tf.get_variable(
                    'w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable(
                    'b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                out = tf.matmul(l1, w2) + b2
            return out

        self.state_this_round = tf.placeholder(
            tf.float32, [None, self.n_features], name='state_this_round')
        self.state_next_round = tf.placeholder(
            tf.float32, [None, self.n_features], name='state_next_round')
        self.reward = tf.placeholder(tf.float32, [None, ], name='reward')
        self.action = tf.placeholder(tf.int32, [None, ], name='action')

        with tf.variable_scope('evaluate_net'):
            c_names, n_l1, w_initializer, b_initializer = [
                'evaluate_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 10, tf.random_normal_initializer(0, 0.3), tf.constant_initializer(0.1)
            self.q_eval = build_layers(
                self.state_this_round, c_names, n_l1, w_initializer, b_initializer)

        with tf.variable_scope('target_net'):
            c_names, n_l1, w_initializer, b_initializer = [
                'target_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 10, tf.random_normal_initializer(0, 0.3), tf.constant_initializer(0.1)
            self.q_next = build_layers(
                self.state_next_round, c_names, n_l1, w_initializer, b_initializer)

        with tf.variable_scope('q_target'):
            q_target = self.reward + self.gamma * \
                tf.reduce_max(self.q_next, axis=1, name='Qmax_s_')
            self.q_target = tf.stop_gradient(q_target)
        with tf.variable_scope('q_eval'):
            action_one_hot = tf.one_hot(
                self.action, depth=self.n_actions, dtype=tf.float32)
            self.q_eval_wrt_action = tf.reduce_sum(
                self.q_eval * action_one_hot, axis=1)

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(
                self.q_target, self.q_eval_wrt_action, name='TD_error'))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(
                self.lr).minimize(self.loss)

    def store_transition(self, state_this_round, action, reward, state_next_round):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack(
            [state_this_round, [action, reward], state_next_round])
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, observation):
        observation = observation[np.newaxis, :]
        if np.random.uniform() < self.epsilon:
            action_value = self.sess.run(self.q_eval, feed_dict={
                                         self.state_this_round: observation})
            action = np.argmax(action_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def _replace_target_params(self):
        target_params = tf.get_collection('target_net_params')
        evaluate_params = tf.get_collection('evaluate_net_params')
        self.sess.run([tf.assign(t, e)
                       for t, e in zip(target_params, evaluate_params)])

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self._replace_target_params()
            print('\nreplace params.')

        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(
                self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(
                self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        _, cost = self.sess.run(
            [self._train_op, self.loss],
            feed_dict={
                self.state_this_round: batch_memory[:, :self.n_features],
                self.action: batch_memory[:, self.n_features],
                self.reward: batch_memory[:, self.n_features + 1],
                self.state_next_round: batch_memory[:, -self.n_features]
            })
        self.cost_his.append(cost)

        self.epsilon = self.epsilon + \
            self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()

if __name__ == '__main__':
    DQN = DeepQNetwork(3,4, output_graph=True)

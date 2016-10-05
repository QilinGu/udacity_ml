#
# Reinforcement learning agent for the "driverless car" using Tensorflow
#

import tensorflow as tf
import numpy as np
import random
from carmunk import Game

# Network parameters
n_hidden_1 = 96
n_hidden_2 = 384
n_hidden_3 = 48
n_min_epsilon = 0.01
n_input = 3
n_actions = 3
n_gamma = 0.99
n_observe = 10000
n_memory_size = 100000
n_network_update_frames = 10000
n_history = 4
n_input = n_input * n_history
n_batch_size = 32

class SQN:

    # Shallow Q-Learning network, as opposed to "deep"
    
    def __init__(self):
        # Tf graph input
        self.x = tf.placeholder("float", [None, n_input])

        # Store layers weight & bias, initialize with white noise near 0
        self.weights = {
            'h1': tf.Variable(0.1*tf.random_normal([n_input, n_hidden_1])),
            'h2': tf.Variable(0.1*tf.random_normal([n_hidden_1, n_hidden_2])),
            'h3': tf.Variable(0.1*tf.random_normal([n_hidden_2, n_hidden_3])),
            'out': tf.Variable(0.1*tf.random_normal([n_hidden_3, n_actions]))
        }
        self.biases = {
            'b1': tf.Variable(0.1*tf.random_normal([n_hidden_1])),
            'b2': tf.Variable(0.1*tf.random_normal([n_hidden_2])),
            'b3': tf.Variable(0.1*tf.random_normal([n_hidden_3])),
            'out': tf.Variable(0.1*tf.random_normal([n_actions]))
        }
        self.q_value = self.build_perceptron(self.x, self.weights, self.biases)
        self.q_action = tf.argmax(self.q_value, dimension=1)
        self.optim = self.build_optimizer()

    def copy_sqn(self, session, sqn):
        for key in self.weights.keys():
            session.run(tf.assign(self.weights[key], sqn.weights[key]))
        for key in self.biases.keys():
            session.run(tf.assign(self.biases[key], sqn.biases[key]))

    def build_perceptron(self, x, weights, biases):
        layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['h1']), biases['b1']))
        layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']))
        layer_3 = tf.nn.relu(tf.add(tf.matmul(layer_2, weights['h3']), biases['b3']))
        return tf.add(tf.matmul(layer_3, weights['out']), biases['out'])

    def build_optimizer(self):
        self.target_qt = tf.placeholder('float32', [None])
        self.target_action = tf.placeholder('int32', [None])
        action_one_hot = tf.one_hot(self.target_action, n_actions, 1.0, 0.0)
        q_acted = tf.reduce_sum(self.q_value * action_one_hot, reduction_indices=1)
        self.loss = tf.reduce_mean(tf.square(self.target_qt - q_acted), name='loss')
        return tf.train.RMSPropOptimizer(0.00025,0.99,0.0,1e-6).minimize(self.loss)

    def predict(self, session, states):
        return session.run(self.q_value, feed_dict={self.x: states})

    def learn(self, session, target_network, samples):
        a = np.array(samples)
        X = np.stack(a[:,0])
        actions = np.stack(a[:,1])
        rewards = np.stack(a[:,2])
        X_t1 = np.stack(a[:,3])
        dead_ends = np.stack(a[:,4])
        q_t1 = target_network.predict(session, X_t1)
        max_q_t1 = np.max(q_t1, axis=1)*(1-dead_ends)
        target_q_t1 = n_gamma * max_q_t1  + rewards
        inputs = {self.target_qt: target_q_t1,
                  self.target_action: actions,
                  self.x: X}
        _, q_t, loss = session.run([self.optim, self.q_value, self.loss], inputs)
        return q_t, loss

# Construct model

def prepare_frame(frame):
    # frame is sonar1, sonar2, sonar3
    # sonar ranges 1 to 40
    #
    # we want normalized values between -1 and 1
    #
    return (np.array(frame)-20.0)/20.0

class Learner:

    def __init__(self):
        self.s = tf.Session()
        self.q_train = SQN()
        self.q_target = SQN()
        self.games_played = 0
        self.min_epsilon = n_min_epsilon
        self.reset()

    def mute(self):
        toggle = not self.g.draw_screen
        self.g.draw_screen = toggle
        self.g.show_sensors = toggle
        return not toggle

    def reset(self):
        self.g = Game(0.4)
        self.t = 0
        self.learning_step = 0
        self.epsilon = 1.0
        self.replay = []
        self.losses = []
        self.q_t = None
        self.s_t = None
        self.a_t = None
        self.r_t = 0
        self.s_t1 = None
        self.q_t1 = None
        self.terminal = False
        self.init = tf.initialize_all_variables()
        self.s.run(self.init)
        _, frame, terminal = self.g.step(0)
        frame = prepare_frame(frame)
        self.frames = [frame for i in range(n_history)]

    def guess_actions(self):
        self.s_t = np.ravel(np.array(self.frames))  #state
        self.q_t = self.q_train.predict(self.s, np.array([self.s_t]))[0]

    def choose_action(self):
        # choose an action
        if random.random() < self.epsilon or self.t < n_observe:
            self.a_t = np.random.randint(0,3)
            self.g.state.hud = "*"+str(self.g.total_reward)
        else:
            self.a_t = self.q_t.argmax() # best action index
            self.g.state.hud = str(self.g.total_reward)
        if self.epsilon > self.min_epsilon:
            self.epsilon *= 0.95

    def act_and_observe(self):
        # take action, get reward, new frame
        self.r_t, frame_t1, self.terminal = self.g.step(self.a_t) 
        frame_t1 = prepare_frame(frame_t1)
        self.s_t1 = self.frames[1:]
        self.s_t1.append(frame_t1)
        self.frames = self.s_t1
        self.s_t1 = np.ravel(np.array(self.frames))

    def remember_for_later(self):
        self.replay.append([self.s_t, self.a_t, self.r_t, self.s_t1, self.terminal*1, np.max(self.q_t)])
        if (len(self.replay) > n_memory_size):
            self.replay.pop(0)
        if self.terminal:
            self.g.reset()
            if self.epsilon < self.min_epsilon:
                self.epsilon = self.min_epsilon
            self.games_played += 1

    def learn(self):
        if self.t > n_observe:
            self.learning_step += 1
            q_t, loss = self.q_train.learn(self.s, self.q_target, random.sample(self.replay, n_batch_size))
            self.losses.append(loss)
            if (self.learning_step % n_network_update_frames) == 0:
                print "Games played ", self.games_played
                self.q_target.copy_sqn(self.s, self.q_train)

    def step(self):
        # 1 frame
        self.t += 1
        self.guess_actions()
        self.choose_action()
        self.act_and_observe()
        self.remember_for_later()
        self.learn()

    def demo(self, n=1000):
        # 1k frames
        for i in range(n):
            self.step()

    def cycle(self):
        # 100k frames
        loss_data = []
        for i in range(10):
            self.losses = []
            for i in range(10000):
                self.step()
            print "t=", self.t
            if len(self.losses) > 0:
                these = [np.mean(self.losses), np.std(self.losses), np.min(self.losses), np.max(self.losses)]
                print these
                loss_data.append(these)
        return loss_data

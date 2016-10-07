#
# Reinforcement learning agent for the "driverless car" using Tensorflow
#

import tensorflow as tf
import numpy as np
import random
from carmunk import Game

# Network parameters

n_hidden_1 = 128
n_hidden_2 = 256
n_hidden_3 = 128

n_min_epsilon = 0.01
n_input = 3
n_actions = 3
n_gamma = 0.90
n_observe = 6400
n_explore = 100000
n_memory_size = 32000
n_network_update_frames = 10000
n_history = 1
n_input = n_input * n_history
n_batch_size = 32

class SQN:

    # Shallow Q-Learning network, as opposed to "deep"
    
    def __init__(self, name='q_value'):
        # Tf graph input
        self.name = name
        self.summaries = None
        self.optim = None
        self.y_prime = None
        self.x = tf.placeholder("float", [None, n_input], name="X")

        # Store layers weight & bias, initialize with white noise near 0
        self.weights = [['w1', [n_input, n_hidden_1]],
                        ['w2', [n_hidden_1, n_hidden_2]],
                        ['w3', [n_hidden_2, n_hidden_3]],
                        ['w_out', [n_hidden_3, n_actions]]]
        self.weights = self.make_vars(self.weights, 0.1)

        self.biases =  [['b1', [n_hidden_1]],
                        ['b2', [n_hidden_2]],
                        ['b3', [n_hidden_3]],
                        ['b_out', [n_actions]]]
        self.biases = self.make_vars(self.biases, 0.01)
        self.q_value = self.build_perceptron()
        self.q_action = tf.argmax(self.q_value, dimension=1)

    def make_vars(self, spec, scale):
        vars = {}
        for name, shape in spec:
            vars[name] = tf.Variable(scale*tf.random_normal(shape), name=name)
        return vars

    def copy_sqn(self, session, sqn):
        for key in self.weights.keys():
            session.run(tf.assign(self.weights[key], sqn.weights[key]))
        for key in self.biases.keys():
            session.run(tf.assign(self.biases[key], sqn.biases[key]))

    def build_perceptron(self):
        n = self.name
        w = self.weights
        b = self.biases
        layer_1 = tf.nn.relu(tf.add(tf.matmul(self.x, w['w1']), b['b1']))
        layer_1 = tf.nn.dropout(layer_1, 0.8)
        layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, w['w2']), b['b2']))
        layer_2 = tf.nn.dropout(layer_2, 0.8)
        layer_3 = tf.nn.relu(tf.add(tf.matmul(layer_2, w['w3']), b['b3']))
        layer_3 = tf.nn.dropout(layer_3, 0.8)
        result = tf.add(tf.matmul(layer_3, w['w_out']), b['b_out'])
        return result

    def build_optimizer(self):
        self.y_prime = tf.placeholder('float32', [None], name='y_prime')
        self.action = tf.placeholder('int32', [None], name='action')
        action_one_hot = tf.one_hot(self.action, n_actions, 1.0, 0.0)
        y = tf.reduce_sum(self.q_value * action_one_hot, reduction_indices=1)
        self.loss = tf.reduce_mean(tf.square(self.y_prime - y))
        tf.scalar_summary('loss', self.loss)
        self.optim = tf.train.RMSPropOptimizer(0.00025,0.99,0.0,1e-6).minimize(self.loss)

    def predict(self, session, states):
        feed = {self.x: states}
        return session.run(self.q_value, feed_dict=feed)

    def learn(self, session, target_network, samples):
        a = np.array(samples)
        X = np.stack(a[:,0])
        actions = np.stack(a[:,1])
        rewards = np.stack(a[:,2])
        X_t1 = np.stack(a[:,3])
        dead_ends = np.stack(a[:,4])
        max_q_t1 = np.max(target_network.predict(session, X_t1), axis=1)*(1-dead_ends)
        y_prime = n_gamma * max_q_t1  + rewards
        inputs = {self.y_prime: y_prime,
                  self.action: actions,
                  self.x: X}
        summary, _, q_t, loss = session.run([self.summaries, self.optim, self.q_value, self.loss], inputs)
        return summary, q_t, loss

# Construct model

def prepare_frame(frame):
    # frame is sonar1, sonar2, sonar3
    # sonar ranges 1 to 40
    #
    # we want normalized values between 0 and 1
    #
    return np.array(frame)/20.0

class Learner:

    def __init__(self):
        self.s = tf.Session()
        self.q_train = SQN('q_train')
        self.q_train.build_optimizer()
        self.q_target = SQN('q_target')
        self.games_played = 0
        self.min_epsilon = n_min_epsilon
        self.tf_best_q = tf.Variable(0, name="q_best")
        tf.scalar_summary('q_max', self.tf_best_q)
        self.reset()

    def mute(self):
        toggle = not self.g.draw_screen
        self.g.draw_screen = toggle
        self.g.show_sensors = toggle
        return not toggle

    def start_logging(self):
        self.train_writer = tf.train.SummaryWriter('./train', self.s.graph)

    def stop_logging(self):
        self.train_writer.close()

    def log(self, summary):
        self.train_writer.add_summary(summary, self.t)

    def reset(self):
        self.g = Game(0.4)
        self.t = 0
        self.learning_step = 0
        self.epsilon = 1.0
        self.replay = []
        self.losses = []
        self.games = []
        self.best_q = -500
        self.q_t = None
        self.s_t = None
        self.a_t = None
        self.r_t = 0
        self.s_t1 = None
        self.q_t1 = None
        self.terminal = False
        # enable logging
        self.q_train.summaries = self.q_target.summaries = self.summaries = tf.merge_all_summaries()
        self.init = tf.initialize_all_variables()
        self.s.run(self.init)
        self.start_logging()
        _, frame, terminal = self.g.step(0)
        frame = prepare_frame(frame)
        self.frames = [frame for i in range(n_history)]

    def guess_actions(self):
        self.s_t = np.ravel(np.array(self.frames))  #state
        self.q_t = self.q_train.predict(self.s, np.array([self.s_t]))[0]
        # log our best q_value to date
        self.best_q = max(np.max(self.q_t), self.best_q)
        summary, _ = self.s.run([self.summaries, tf.assign(self.tf_best_q, self.best_q)],
            feed_dict={self.q_train.x: [[0,0,0]], self.q_train.y_prime: [0], self.q_train.action: [0]})
        self.log(summary)

    def choose_action(self):
        # choose an action
        if random.random() < self.epsilon or self.t < n_observe:
            self.a_t = np.random.randint(0,3)
            self.g.state.hud = "*"+str(self.g.total_reward)
        else:
            self.a_t = self.q_t.argmax() # best action index

            self.g.state.hud = str(self.g.total_reward)
        if self.epsilon > self.min_epsilon and self.t > n_observe:
	    self.epsilon -= (1.0 - self.min_epsilon)/(n_explore*1.0)

    def act_and_observe(self):
        # take action, get reward, new frame
        self.r_t, frame_t1, self.terminal = self.g.step(self.a_t) 
        frame_t1 = prepare_frame(frame_t1)
        self.s_t1 = self.frames[1:]
        self.s_t1.append(frame_t1)
        self.frames = self.s_t1
        self.s_t1 = np.ravel(np.array(self.frames))

    def remember_for_later(self):
#        self.r_t = min(10,max(-10, self.r_t))
        self.replay.append([self.s_t, self.a_t, self.r_t, self.s_t1, self.terminal*1, np.max(self.q_t)])
        if (len(self.replay) > n_memory_size):
            self.replay.pop(0)
        if self.terminal:
            self.games.append(self.g.state.num_steps)
            self.g.total_reward = 0
            self.g.state.num_steps = 0
            self.games_played += 1

    def learn_by_replay(self):
        if self.t > n_observe:
            self.learning_step += 1
            summary, q_t, loss = self.q_train.learn(self.s, self.q_target, random.sample(self.replay, n_batch_size))
            self.log(summary)
            self.losses.append(loss)
            if (self.learning_step % n_network_update_frames) == 0:
                if len(self.games):
                    print "Games played ", self.games_played
                    print "Epoch Max score", np.max(self.games)
                    print "Epoch Mean score", np.mean(self.games)
                self.games = []
                self.q_target.copy_sqn(self.s, self.q_train)

    def step(self):
        # 1 frame
        self.t += 1
        self.guess_actions()
        self.choose_action()
        self.act_and_observe()
        self.remember_for_later()
        self.learn_by_replay()

    def demo(self, n=1000):
        # 1k frames
        not self.mute() or not self.mute()
        for i in range(n):
            self.step()

    def cycle(self, n=10):
        # 100k frames
        self.mute() or self.mute()
        self.g.reset()
        loss_data = []
        for i in range(n):
            self.losses = []
            for i in range(10000):
                self.step()
            print "t=", self.t
            if len(self.losses) > 0:
                these = [np.mean(self.losses), np.std(self.losses), np.min(self.losses), np.max(self.losses)]
                print these
                loss_data.append(these)
        return loss_data

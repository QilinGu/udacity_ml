#
# Reinforcement learning agent for the "driverless car" using Tensorflow
#
# @scottpenberthy
# November 1, 2016
#

import tensorflow as tf
import numpy as np
import random
from carmunk import Game
import pygame
import math

# Network parameters

n_hidden = [32] + [64]*6 + [32]
n_min_epsilon = 0.01
n_input = 6
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
    
    def __init__(self, name='q_value', track=False):
        self.name = name
        self.summaries = None
        self.optim = None
        self.y_prime = None
        w = []
        b = []
        self.x = tf.placeholder("float", [None, n_input], name="X")

        # Store layers weight & bias, initialize with white noise near 0
        dims = [n_input]+n_hidden+[n_actions]
        for i in range(len(dims)-1):
            _i = str(i+1)
            w += [['w'+_i, dims[i:i+2]]]
            b += [['b'+_i, dims[i+1:i+2]]]
        self.weights = self.make_vars(w, track=track)
        self.biases = self.make_vars(b, constant=True, track=track)
        self.q_value = self.build_perceptron()
        if track:
            tf.histogram_summary(self.name + "/q_value", self.q_value)
        self.q_action = tf.argmax(self.q_value, dimension=1)
        self.q_max = tf.Variable(0.0, name="q_max/" + name)
        self.q_max_val = 0.0

    def make_vars(self, spec, constant=False, track=False):
        # Create "variables" in Tensorflow that we'll later
        # assign to values.   We use this handy routine
        # for creating our weights and biases.  We need to pull
        # them out as variables to allow for copying from
        # our smarter, training network to the operational, 
        # target network every n_network_update_frames frames.
        vars = {}
        for name, shape in spec:
            if constant:
                vars[name] = tf.Variable(tf.constant(0.01, shape=shape), name=name)
            else:
                vars[name] = tf.Variable(tf.truncated_normal(shape, stddev=0.01), name=name)
            if track:
                tf.histogram_summary(self.name + "/" + name, vars[name])
        return vars

    def copy_sqn(self, session, sqn):
        #
        # Load our weights and biaes from the trained network to the operational
        # target network.  We do this with an tensorflow assignment. 
        #
        for key in self.weights.keys():
            session.run(tf.assign(self.weights[key], sqn.weights[key]))
        for key in self.biases.keys():
            session.run(tf.assign(self.biases[key], sqn.biases[key]))

    def build_perceptron(self):
        #
        # Build a fully connected network whose output computes a regression.
        # The hidden layers all use a 20% dropout filter and a nonliner rectifier
        # for activation.  The final layer uses a linear combination to feed
        # the final layer output.  We store the weights and biases using 
        # a naming convention of wi and bi for layer i.
        #
        w = self.weights
        b = self.biases 
        layers = [tf.nn.relu(tf.add(tf.matmul(self.x, w['w1']), b['b1']))]
        for i in range(1, len(n_hidden)):
            _i = str(i+1)
            layers[i-1] = tf.nn.dropout(layers[i-1], 0.8)
            layers += [tf.nn.relu(tf.add(tf.matmul(layers[i-1], w['w'+_i]),b['b'+_i]))]
        nth = str(len(n_hidden)+1)
        result = tf.add(tf.matmul(layers[-1:][0], w['w'+nth]), b['b'+nth])
        return result

    def build_optimizer(self):
        #
        # Build our back propogation step.  We optimize by minimizing
        # squared error between our target prediction for Q() and what
        # our training network is currently predicting for Q().
        #
        # This implements Bellman's equation
        #
        # Q*(s,a) = Rt + gamma * Max(Q(s',a'))
        #
        # When we take action a in state s, our expected longterm reward
        # is the reward we receive for that first action a, then the
        # maximum longterm reward available to us in the new state s',
        # from all actions a' we could take in s'.
        #
        self.y_prime = tf.placeholder('float32', [None], name='y_prime')
        self.action = tf.placeholder('int32', [None], name='action')
        action_one_hot = tf.one_hot(self.action, n_actions, 1.0, 0.0)
        y = tf.reduce_sum(self.q_value * action_one_hot, reduction_indices=1)
        self.loss = tf.reduce_mean(tf.square(self.y_prime - y))
        tf.scalar_summary('loss', self.loss)
        tf.scalar_summary('qmax', self.q_max)
        self.optim = tf.train.RMSPropOptimizer(0.00025,0.99,0.0,1e-6).minimize(self.loss)

    def predict(self, session, states):
        #
        # Feed forward a set of states through our network and
        # extract the predicted Q values.  Note that we also use the trick
        # of tracking a Python variable across network evaluations by 
        # jamming it into the state at every iteration (self.q_max).
        #
        feed = {self.x: states, self.q_max: self.q_max_val}
        qv = session.run(self.q_value, feed_dict=feed)
        self.q_max_val = max(np.max(qv), self.q_max_val)
        return qv

    def learn(self, session, target_network, samples):
        #
        # Perform a back-propagation.
        #
        a = np.array(samples)
        X = np.stack(a[:,0])
        actions = np.stack(a[:,1])
        rewards = np.stack(a[:,2])
        X_t1 = np.stack(a[:,3])
        dead_ends = np.stack(a[:,4])
        max_q_t1 = np.max(target_network.predict(session, X_t1), axis=1)*(1-dead_ends)
        self.q_max_val = max(np.max(max_q_t1), self.q_max_val)
        y_prime = rewards + n_gamma * max_q_t1
        inputs = {self.y_prime: y_prime,
                  self.action: actions,
                  self.q_max: self.q_max_val,
                  self.x: X}
        summary, _, q_t, loss = session.run([self.summaries, self.optim, self.q_value, self.loss], inputs)
        return summary, q_t, loss


# Construct model

def prepare_frame(frame):
    # frame is sonar1, sonar2, sonar3, x, y, theta
    # sonar ranges 0 to 40
    # x and y range 0 to 1
    # theta ranges 0 to two pi
    #
    # we want normalized values between 0 and 1
    #
    s1, s2, s3, x, y, theta = frame
    return [s1/40.0, s2/40.0, s3/40.0, x, y, theta/(2*math.pi)]

class Learner:

    # This is our reinforcement learner that uses
    # two SQN networks to update Q values.
    # 
    # We first wait and take random actions for
    # n_observe frames.  This helps us fill up 
    # our event memory.  Next, we continue to take
    # random actions as we gradually anneal self.epsilon
    # down to self.min_epsilon over n_explore frames.

    def __init__(self, plotting_only=False):
        self.s = tf.Session()
        self.q_train = SQN('q_train', True)
        self.q_train.build_optimizer()
        self.q_target = SQN('q_target')
        self.games_played = 0
        self.min_epsilon = n_min_epsilon
        self.score = tf.Variable(0.0, name="score")
        self.score_val = 0.0
        tf.scalar_summary('score', self.score)
        self.reset(plotting_only)

    def mute(self):
        # Toggle the visual display on or off for performance reasons.
        toggle = not self.g.draw_screen
        self.g.draw_screen = toggle
        self.g.show_sensors = toggle
        return not toggle

    def start_logging(self):
        #
        # This is how we log information for use in Tensorboard.  
        #
        self.train_writer = tf.train.SummaryWriter('./train', self.s.graph)

    def stop_logging(self):
        # This flushes whatever remains in memory to disk for Tensorboard.
        self.train_writer.close()

    def log(self, summary):
        # This updates the logfile with output from the latest
        # iteration, which Tensorboard calls a "summary". 
        # 
        # Not to be confused with a "shrubbery."
        #
        self.train_writer.add_summary(summary, self.t)

    def init_common(self):
        # initialize variables common to training and testing
        self.t = 0
        self.learning_step = 0
        self.replay = []
        self.losses = []
        self.games = []
        self.q_t = None
        self.s_t = None
        self.a_t = None
        self.r_t = 0
        self.s_t1 = None
        self.q_t1 = None
        self.terminal = False
        self.test_mode = False
        self.baseline = False
        # enable logging
        self.q_train.summaries = self.q_target.summaries = self.summaries = tf.merge_all_summaries()

    def init_for_training(self):
        #
        # This is kind of bogus, but I had to tease out
        # init conditions for when we want to train with
        # null values, epsilon, and the works.
        #
        self.epsilon = 1.0
        self.init = tf.initialize_all_variables()
        self.s.run(self.init)

    def init_for_testing(self):
        # 
        # Here we squash the learner hyperparameters
        # to force our agent to use its own prediction
        # at every iteration.
        #
        self.baseline = False
        self.epsilon = 0
        self.learning_step = 0
        self.min_epsilon = 0
        self.t = n_observe+1

    def reset(self, plotting_only = False, initvars=True):
        #
        # Go to a virgin state.  The plotting_only flag is needed
        # to avoid conflicts between PyGame and MatPlotLib which
        # I needed to showcase final results.  The initvars
        # flag is used when we want to evaluate a model for
        # performance - we don't want to wipe out weights and
        # biases we've loaded from one of our models.
        #
        # See plotting.py
        #
        if not plotting_only:
            self.g = Game(1.0)
        self.init_common()
        if initvars:
           self.init_for_training()
        else:
            self.init_for_testing()
        if not plotting_only:
            self.init_game()

    def init_game(self):
        #
        # Start our PyGame simulator, grab
        # a frame and make it or first state.
        #
        # Note you can change n_history to the number of 
        # recent events you want to keep as part of the state
        # as in the Deep Q Learning paper.  I ended up with 1 after
        # trying many combo's.
        #
        self.start_logging()
        _, frame, terminal = self.g.step(0)
        frame = prepare_frame(frame)
        self.frames = [frame for i in range(n_history)]

    def guess_actions(self):
        #
        # Use the feed-forward network to compute Q(s,a) at time t for all a, at once.
        #
        self.s_t = np.ravel(np.array(self.frames))  #state
        self.q_t = self.q_target.predict(self.s, np.array([self.s_t]))[0]

    def choose_action(self):
        # choose an action
        #
        # If we want a random baseline, or are learning and epsilon isn't 
        # degraded, or we haven't finished observing, then pick a random act.
        if self.baseline or (random.random() < self.epsilon) or (self.t < n_observe):
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

    def track_top_score(self):
        # as it saysa.  We track in Python, then stuff the value into Tensorflow for
        # Tensorboard to pick up.  We also track the number of steps we were able
        # to achieve to print out basic progress during an epoch (say 10,000 frames).
        #
        self.games.append(self.g.state.num_steps)
        self.score_val = max(self.score_val, self.games[-1])
        self.s.run(tf.assign(self.score, self.score_val))

    def remember_for_later(self):
        # Add to our local memory from which we train our network. This offline training
        # from recent memory is based on actual biology of the hippocampus.
#        self.r_t = min(10,max(-10, self.r_t))
        self.replay.append([self.s_t, self.a_t, self.r_t, self.s_t1, self.terminal*1, np.max(self.q_t)])
        if (len(self.replay) > n_memory_size):
            self.replay.pop(0)
        if self.terminal:
            self.track_top_score()
            self.g.total_reward = 0
            self.g.state.num_steps = 0
            self.games_played += 1
            
    def get_batch(self):
        # 
        # Fetch a random batch of states to learn from.  Actually, its
        # not so random here.  I make sure half of the states are penalties,
        # or as many as we have, then fill the rest with good ones.
        #
        a = np.array(self.replay)
        goofs = a[a[:,2] < 0]
        oops = random.sample(goofs, min(len(goofs),n_batch_size/2))
        yay = a[a[:,2] >= 0]
        ok = random.sample(yay, min(n_batch_size - len(oops), len(yay)))
        return np.concatenate((ok,oops))

    def show_epoch_stats(self):
        if len(self.games):
            print "Games played ", self.games_played
            print "Epoch Max score", np.max(self.games)
            print "Epoch Mean score", np.mean(self.games)

    def learn_by_replay(self):
        # As it says.  Grab a batch, see how we're off from the target,
        # and update our weights.
        if self.t > n_observe:
            self.learning_step += 1
            summary, q_t, loss = self.q_train.learn(self.s, self.q_target, self.get_batch())
            if (self.learning_step % 100) == 99:
                self.log(summary)
                self.losses.append(loss)
            if (self.learning_step % n_network_update_frames) == 0:
                self.show_epoch_stats()
                if not self.test_mode:
                    self.games = []
                self.q_target.copy_sqn(self.s, self.q_train)

    def step(self):
        # 1 frame
        self.t += 1
        self.guess_actions()
        self.choose_action()
        self.act_and_observe()
        self.remember_for_later()
        if not self.test_mode:
            self.learn_by_replay()

    def demo(self, test=True, n=1000):
        # Perform 1000 frames with a live demo.
        not self.mute() or not self.mute()
        if test:
            self.test_mode = True
            self.games = []
        for i in range(n):
            self.step()
        if test:
            self.test_mode = False
            self.show_epoch_stats()

    def debug(self):
        #
        # Enter a continuous loop that looks for any keyboard
        # or mouse activity, printing out our Q values at
        # each iteration.  I use this to test the network
        # and see what's coming from the simulator.
        #
        ok = True
        while ok:
            self.step()
            print self.t, self.q_t.tolist(), 'R=', self.r_t
            pygame.event.get()
            pygame.event.wait()

    def explore(self):
        #
        # This is similar to debug, expect we print out
        # the raw state information instead.
        #
        ok = True
        while ok:
            self.step()
            print self.t, self.s_t.tolist(), 'R=', self.r_t
            pygame.event.get()
            pygame.event.wait()

    def evaluate(self, initvars=True):
        #
        # Create an even playing field for evaluting a model.
        # Run it for 1000 games and print out basic statistics.
        #
        self.games_played = 0
        random.seed(0)
        np.random.seed(seed=0)
        tf.set_random_seed(0)
        self.reset(initvars=initvars)
        self.games = []
        self.test_mode = True
        while not self.mute():
            pass
        while self.games_played < 1000:
            self.step()
            if (self.games_played > 0) and (self.t % 1000 == 0):
                print self.games_played, "games played with top score", np.max(self.games)
        self.test_mode = False
        a = np.array(self.games)
        std1 = np.std(a)
        mean = np.mean(a)
        low = np.min(a)
        hi = np.max(a)
        print "Mean score:", mean
        print "Std dev:", std1
        print "Low:", low
        print "Hi:", hi
        self.reset()


    def cycle(self, n=10):
        # 
        # This is the core of the trainer.  It runs 10,000 steps 
        # and trains merrily along without the graphic visualization
        # to slow it down.
        # 
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

    def load_winner(self):
        #
        # Load the winning, saved model from the final project.
        #
        self.saver = tf.train.Saver()
        self.saver.restore(self.s,"models/narrow-deep-pipe.ckpt")

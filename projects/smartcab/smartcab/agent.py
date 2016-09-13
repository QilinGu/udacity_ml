import random
import numpy as np
import pandas as pd
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        self.destination = None
        
        # TODO: Initialize any additional variables here
        self.alpha = 0.2
        self.gamma = 0.8
        self.qdict = {}
        self.rewards = [0.0]

    def reset_state(self):
        self.state = {}
        for x in self.env.valid_inputs:
            self.state[x] = None

    def show_q(self):
        print "*** QDICT ***"
        print sorted(self.state.keys())
        for state in self.qdict.keys():
            print state
            print self.qdict[state]
        print

    def show_state(self, t):
        print "My state at time", t, ": "
        for key in self.state.iterkeys():
            print key, self.state[key]
        print
        
    def init_q(self, index):
        # Return our initial guess for our "quaity" function q(s,a)
        # that shows the score for a state s (our current state) taking
        # an action 'a'.
        #
        # We store q as a dictionary whose keys are the ordered
        # key values of state.  The entries are dictionaries, one entry
        # for each possible action.  The entry values are numeric.
        #
        # We use a random distribution weighted by 0.2, giving us some
        # randomness and offsetting bad moves a bit.  Without this, we
        # get stuck at "None" quite often waiting out the clock.  We need
        # to experiment more, like we all do when learning.  I chose this
        # approach vs introducing a random action epsilon percent of the time.
        # Works better.
        #
        return {action:0.2*random.random() for action in self.env.valid_actions}

    def state_index(self):
        # Turn our current state into an index for our quality function q(s,a).
        return tuple([self.state[x] for x in sorted(self.state.keys())])

    def q(self):
        #
        # Return the table of actions for the quality function
        # at the current state.  If they don't yet exist, create
        # them using our initial distribution of guesses in init_q.
        #
        ix = self.state_index()
        v = self.qdict.get(ix, None) 
        if v is None:
           v = self.init_q(ix)
           self.qdict[ix] = v
           print len(self.qdict.keys()), " states created: "
           self.show_state(0)
        return v

    def q_value(self, action):
        #
        # The quality function q(s,a):  given our current state
        # s, what is the quality score of taking action a?
        #
        return self.q()[action]

    def policy_move(self):
       #
       # The "policy" for predicting an action is the argmax_a Q(s,a). 
       # Return that value for the current state.
       #
       q_ = self.q()
       return max(iter(q_), key = lambda action: q_[action])

    def max_q(self):
       #
       # Return the maximum quality value we can achieve from 
       # our current state.  Shuffle to give us some randomness
       # when we face equal q values.
       #
       q_ = self.q()
       actions = list(q_.keys())
       random.shuffle(actions)  # shuffle in lieu of epsilon
       a = max(actions, key = lambda action: q_[action])
       return q_[a]

    def lerp(self, old, newval):
        #
        # linearly interpolate on alpha, implementing <-- alpha --
        #
        return self.alpha*newval + (1-self.alpha)*old

    def reset(self, destination=None):
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.planner.route_to(destination)
        self.destination = destination
        self.reset_state()

    def update_state(self,t):
        self.reset_state()  # markov... polo!
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        self.state['waypoint'] = self.next_waypoint
        inputs = self.env.sense(self)
        for x in self.env.valid_inputs:
            self.state[x] = inputs[x]

    def update(self, t):
        # Gather inputs
        self.update_state(t)

        # TODO: Select action according to your policy
        q_hat = self.q()
        action = self.policy_move()

        # Execute action and get reward
        reward = self.env.act(self, action)
        self.rewards.append(self.rewards[-1]+reward)

        # TODO: Learn policy based on state, action, reward
        q_hat[action] = self.lerp(q_hat[action], reward + self.gamma * self.max_q())


import matplotlib.pyplot as plt

def run():
    """Run the agent for a finite number of trials."""
    data = []

    for alpha in [0.2, 0.4, 0.6, 0.8]:
        for gamma in [0.2, 0.4, 0.6, 0.8]:

            # Set up environment and agent
            random.seed(0)
            e = Environment()  # create environment (also adds some dummy traffic)
            a = e.create_agent(LearningAgent)  # create agent
            a.alpha = alpha
            a.gamma = gamma
            e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track

            # Now simulate it
            sim = Simulator(e, update_delay=0, display=False)
            # NOTE: To speed up simulation, reduce update_delay and/or set display=False

            trials = sim.run(n_trials=500)  # run for a specified number of trials
            # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line
            data.append([alpha, gamma, trials, a.rewards, len(a.qdict.keys())])
    return data

def show_data(data):
    for entry in data:
        alpha = entry[0]
        gamma = entry[1]

        trials = entry[2]
        win = [trials[i][0] for i in range(0,len(trials))]
        steps = [trials[i][1] for i in range(0,len(trials))]
        score = [trials[i][2] for i in range(0,len(trials))]
        penalty = [trials[i][3] for i in range(0,len(trials))]
        wins = sum(win)

        print "alpha ", alpha, "gamma", gamma, "wins", wins, "states", entry[4]
        df = pd.DataFrame({'steps': steps, 'score': score, 'penalty': penalty})
        print df.describe()
        print

if __name__ == '__main__':
    show_data(run())

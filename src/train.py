from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient

import sys
import os
import datetime
import pickle

import FQI as FQI
import utils as utils

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!
class ProjectAgent:
    def __init__(self):

        # whether to train and save new model or just load an existing model (models/model.pkl)
        self.new = False            
        if( len(sys.argv) > 1 ):        # training a new model is specified through a command line argument: python3 main.py new
            if sys.argv[1] == "new":
                self.new = True

        self.algorithm = "FQI"                      # RL algorithm to train (only relevant if training a new model)
        self.gamma = 0.98                           # discount factor
        self.nb_actions = env.action_space.n        # number of actions

         # if not training a new model, check if the "default" model exists
        if not self.new:                       
            if not os.path.exists( "./models/model.pkl" ):
                self.train = True
            else:
                self.path = "./models/model.pkl"
        
        if self.new:
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")       # time stamp to avoid overwriting models
            self.path = "./models/{}_{}.pkl".format(self.algorithm, timestamp)
            self.train()
            self.save(self.path)

            
    def train(self):
        """
        Train the agent. And save model to self.model
        """
        if self.algorithm == "FQI":
            # The first step is to collect and store a dataset of samples
            S,A,R,S2,D = FQI.collect_samples(env, int(1e4))
            print("nb of collected samples:", S.shape[0])

            # Build the sequence of AVI Q-functions, learned using random forests
            nb_iter = 10
            Qfunctions = FQI.rf_fqi(S, A, R, S2, D, nb_iter, self.nb_actions, self.gamma)

            # Save the last Q-function
            self.model = Qfunctions[-1]

    def act(self, observation, use_random=False):
        a = utils.greedy_action(self.model, observation, self.nb_actions)
        return a

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    def load(self):
        with open(self.path, "rb") as f:
            self.model = pickle.load(f)

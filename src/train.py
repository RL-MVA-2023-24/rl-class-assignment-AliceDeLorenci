import random
import torch
import torch.nn as nn
import numpy as np
import sys
import os
import datetime
import matplotlib.pyplot as plt
from copy import deepcopy
import json
import inspect

from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!
class ProjectAgent:

    def act(self, observation, use_random=False):
        with torch.no_grad():
            Q = self.model(torch.Tensor(observation).unsqueeze(0).to(self.device))
            return torch.argmax(Q).item()

    def save(self, path):
        print("Saving model to", path)
        torch.save(self.model.state_dict(), self.path)

    def load(self):
        self.device = self.test_device

        print("Loading model from", self.path)
        self.model = torch.jit.load(self.path, map_location=self.device)
        self.model.eval()
    
    def __init__(self):
        print("Initializing agent")


        self.path = "./src/models/model_jit.pt" # path to default model

        self.n_actions = env.action_space.n             # number of actions
        self.state_dim = env.observation_space.shape[0] # state space dimension

        self.model = None                               # Q-network
        self.device = None
        self.train_device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # device to use for training
        self.test_device = torch.device("cpu")                                           # device to use for evaluation

        
        # command line argument specifies model to load (otherwise, the default model is loaded):
        # python3 main.py <file_name>
        if( len(sys.argv) > 1 ):        
            self.path = './src/models/'+sys.argv[1]

        # if the model to load does not exist, raise an error
        if not os.path.exists( self.path ):
            raise Exception("Model {} does not exist.".format(self.path))
    
    def greedy_action(self, network, state):
        """
        Select the greedy action according to the Q-network.
        """
        with torch.no_grad():
            Q = network(torch.Tensor(state).unsqueeze(0).to(self.device))
            return torch.argmax(Q).item()
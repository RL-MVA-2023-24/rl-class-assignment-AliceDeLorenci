import random
import torch
import torch.nn as nn
import numpy as np
import sys
import os
import datetime
import matplotlib.pyplot as plt
from copy import deepcopy

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
        self.model = self.network()
        self.model.load_state_dict(torch.load(self.path, map_location=self.device))
        self.model.eval()
    
    def __init__(self):
        print("Initializing agent")

        #### INSTANCE ATTRIBUTES ####
        self.algorithm = "DQNtarget"               
        self.new = False                    # whether to train and save new model or just load an existing model (models/model.<extension>)
        self.path = "./src/models/model.pt" # path to default model
        self.path = './src/models/DQNtarget_20240224-163404.pt'

        self.n_actions = env.action_space.n             # number of actions
        self.state_dim = env.observation_space.shape[0] # state space dimension

        self.model = None                               # Q-network
        self.device = None
        self.train_device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # device to use for training
        self.test_device = torch.device("cpu")                                           # device to use for evaluation
        #############################

        
        # training a new model is specified through a command line argument: python3 main.py new
        if( len(sys.argv) > 1 ):        
            if sys.argv[1] == "new":
                self.new = True

        # if the "default" model does not exist, train a new model
        if not self.new and not os.path.exists( self.path ):
            print("No model found. Training a new model.")
            self.new = True
        
        # train new model
        if self.new:
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")       # time stamp to avoid overwriting models
            self.path = "./src/models/{}_{}.pt".format(self.algorithm, timestamp)
            ep_length, disc_rewards, tot_rewards, V0 = self.train()
            self.save(self.path)

            plt.figure()
            plt.plot(ep_length, label="training episode length")
            plt.plot(tot_rewards, label="MC eval of total reward")
            plt.legend()
            plt.savefig(self.path+'-fig1.png')

            plt.figure()
            plt.plot(disc_rewards, label="MC eval of discounted reward")
            plt.plot(V0, label="average $max_a Q(s_0)$")
            plt.legend()
            plt.savefig(self.path+'-fig2.png')

            
    def network(self, nb_neurons=300):
        """
        Neural Network architecture for DQN.
        """
        DQN = torch.nn.Sequential(nn.Linear(self.state_dim, nb_neurons),
                                  nn.ReLU(),
                                  nn.Linear(nb_neurons, nb_neurons),
                                  nn.ReLU(), 
                                  nn.Linear(nb_neurons, self.n_actions)).to(self.device)
        return DQN

    
    def greedy_action(self, network, state):
        """
        Select the greedy action according to the Q-network.
        """
        with torch.no_grad():
            Q = network(torch.Tensor(state).unsqueeze(0).to(self.device))
            return torch.argmax(Q).item()

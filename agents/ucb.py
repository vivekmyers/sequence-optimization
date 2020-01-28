import numpy as np
from random import *
from models.bayesian import BayesianCNN
from models.auto_cnn import CNN
import agents.base


def UCBAgent(epochs=30):
    '''Constructs agent with a Bayesian CNN, using Thompson sampling with the
    network's uncertainty (over its parameters) to select the highest UCB
    sequences to test in terms of (mu + sigma), and fits the model
    to update the predicted distributions between batches.
    '''

    class Agent(agents.random.RandomAgent(epochs)):

        def __init__(self, *args):
            super().__init__(*args)
            self.model = BayesianCNN(encoder=self.encode, shape=self.shape)
        
        def act(self, seqs):
            mu, sigma = self.model.sample(seqs)
            ucb = mu + sigma
            return list(zip(*sorted(zip(ucb, seqs))[-self.batch:]))[1]

        def observe(self, data):
            super().observe(data)
            self.model.fit(*zip(*self.seen.items()), epochs=epochs)
        
    return Agent

import numpy as np
from random import *
from models.fixed_bayesian import BayesianCNN
from models.auto_cnn import CNN
import agents.random


def ThompsonAgent(epochs=30):
    '''Constructs agent with a Bayesian CNN, using Thompson sampling with the
    network's uncertainty (over its parameters) to select batches, and 
    fitting the model to update the predicted distributions between batches.
    '''

    class Agent(agents.random.RandomAgent(epochs)):

        def __init__(self, *args):
            super().__init__(*args)
            self.model = BayesianCNN(encoder=self.encode, shape=self.shape)
        
        def act(self, seqs):
            mu = self.model.sample(seqs)
            return list(zip(*sorted(zip(mu, seqs))[-self.batch:]))[1]

        def observe(self, data):
            super().observe(data)
            self.model.fit(*zip(*self.seen.items()), epochs=epochs)
        
    return Agent

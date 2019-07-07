import numpy as np
from random import *
import agents.base
from models.gp import GaussianProcess

def GaussianAgent(epochs=200, initial_epochs=None, beta=100., dim=5):
    '''Constructs agent that uses batch version of GP-UCB algorithm to sample
    sequences with a deep kernel gaussian process regression. Beta is the square
    of the weighting of the stdev in selecting actions.
    '''
    if initial_epochs is None:
        initial_epochs = 2 * epochs

    class Agent(agents.base.BaseAgent):

        def __init__(self, *args):
            super().__init__(*args)
            self.model = GaussianProcess(encoder=self.encode, dim=dim, shape=[self.len, 4])
            self.model.fit(*zip(*self.seen.items()), epochs=initial_epochs, 
                                minibatch=min(self.batch, 100))
        
        def act(self, seqs):
            prior = {}
            choices = []
            for i in range(self.batch):
                mu, sigma = self.model.interpolate(seqs, prior)
                idx = np.argmax(mu + np.sqrt(beta) * sigma)
                prior[seqs[idx]] = mu[idx]
                choices.append(idx)
            return np.array(seqs)[choices]

        def observe(self, data):
            super().observe(data)
            self.model.fit(*zip(*self.seen.items()), epochs=epochs, 
                                minibatch=min(self.batch, 100))
        
        def predict(self, seqs):
            return self.model.interpolate(seqs, {})[0]

    return Agent


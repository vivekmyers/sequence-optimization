import numpy as np
from random import *
from models.bayesian import BayesianCNN
import agents.base

def UCBAgent(epochs=50, initial_epochs=None):
    '''Constructs agent with a Bayesian CNN, using Thompson sampling with the
    network's uncertainty (over its parameters) to select the highest UCB
    sequences to test in terms of (mu + sigma), and refits the model
    to update the predicted distributions between batches.
    '''
    if initial_epochs is None:
        initial_epochs = 2 * epochs

    class Agent(agents.base.BaseAgent):

        def __init__(self, *args):
            super().__init__(*args)
            self.model = BayesianCNN(encoder=self.encode, shape=[self.len, 4])
            self.model.fit(*zip(*self.seen.items()), epochs=initial_epochs, minibatch=min(self.batch, 100))
        
        def act(self, seqs):
            mu, sigma = self.model.sample(seqs)
            ucb = mu + sigma
            return list(zip(*sorted(zip(ucb, seqs))[-self.batch:]))[1]

        def observe(self, data):
            super().observe(data)
            self.model.fit(*zip(*self.seen.items()), epochs=epochs, minibatch=min(self.batch, 100))
        
        def predict(self, seqs):
            mu, sigma = self.model.predict(seqs)
            return mu

    return Agent


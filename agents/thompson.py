import numpy as np
from random import *
from models.bayesian import BayesianCNN
import agents.base

def ThompsonAgent(epochs=50, initial_epochs=None):
    '''Constructs agent with a Bayesian CNN, using Thompson sampling with the
    network's uncertainty (over its parameters) to select batches, and 
    refitting the model to update the predicted distributions between batches.
    '''
    if initial_epochs is None:
        initial_epochs = 2 * epochs

    class Agent(agents.base.BaseAgent):

        def __init__(self, *args):
            super().__init__(*args)
            self.model = BayesianCNN(encoder=self.encode, shape=self.shape)
            self.model.fit(*zip(*self.seen.items()), epochs=initial_epochs, minibatch=min(len(self.seen), 100))
        
        def act(self, seqs):
            mu, sigma = self.model.sample(seqs)
            return list(zip(*sorted(zip(mu, seqs))[-self.batch:]))[1]

        def observe(self, data):
            super().observe(data)
            self.model.fit(*zip(*self.seen.items()), epochs=epochs, minibatch=min(len(self.seen), 100))
        
        def predict(self, seqs):
            mu, sigma = self.model.predict(seqs)
            return mu

    return Agent


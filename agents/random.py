import numpy as np
from random import *
import agents.base
from models.cnn import CNN

def RandomAgent(epochs=10, initial_epochs=None):
    '''Constructs agent that uses CNN to predict sequence values.
    Randomly selects new sequences to observe.
    '''
    if initial_epochs is None:
        initial_epochs = epochs // 4

    class Agent(agents.base.BaseAgent):

        def __init__(self, *args):
            super().__init__(*args)
        
        def act(self, seqs):
            return sample(seqs, self.batch)

        def observe(self, data):
            super().observe(data)
        
        def predict(self, seqs):
            model = CNN(encoder=self.encode, shape=self.shape)
            if self.prior: model.fit(*zip(*self.prior.items()), epochs=initial_epochs, minibatch=100)
            if self.seen: model.fit(*zip(*self.seen.items()), epochs=epochs, minibatch=100)
            return model.predict(seqs)

    return Agent


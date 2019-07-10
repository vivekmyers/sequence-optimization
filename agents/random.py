import numpy as np
from random import *
import agents.base
from models.cnn import CNN

def RandomAgent(epochs=10, initial_epochs=None):
    '''Constructs agent with CNN to predict sequence values that trains with each observation.
    Randomly selects new sequences.
    '''
    if initial_epochs is None:
        initial_epochs = 2 * epochs

    class Agent(agents.base.BaseAgent):

        def __init__(self, *args):
            super().__init__(*args)
            self.model = CNN(encoder=self.encode, shape=[self.len, 4])
            self.model.fit(*zip(*self.seen.items()), epochs=initial_epochs, 
                            minibatch=min(len(self.seen), 100))
        
        def act(self, seqs):
            return sample(seqs, self.batch)

        def observe(self, data):
            super().observe(data)
            self.model.fit(*zip(*self.seen.items()), epochs=epochs, 
                            minibatch=min(len(self.seen), 100))
        
        def predict(self, seqs):
            return self.model.predict(seqs)

    return Agent


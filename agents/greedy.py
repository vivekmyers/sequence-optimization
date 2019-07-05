import numpy as np
from random import *
import agents.base
from models.cnn import CNN

def GreedyAgent(epochs=10, initial_epochs=None):
    '''Constructs agent with CNN to predict sequence values that trains with each observation.
    Greedily selects sequences with best predicions.
    '''
    if initial_epochs is None:
        initial_epochs = 2 * epochs

    class Agent(agents.base.BaseAgent):

        def __init__(self, *args):
            super().__init__(*args)
            self.model = CNN(encoder=self.encode, shape=[self.len, 4])
            self.model.fit(*zip(*self.seen.items()), epochs=initial_epochs, 
                            minibatch=min(self.batch, 100))
        
        def act(self, seqs):
            return list(zip(*sorted(zip(self.predict(seqs), seqs))[-self.batch:]))[1]

        def observe(self, data):
            super().observe(data)
            self.model.fit(*zip(*self.seen.items()), epochs=epochs, 
                            minibatch=min(self.batch, 100))
        
        def predict(self, seqs):
            return self.model.predict(seqs)

    return Agent


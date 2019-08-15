import numpy as np
from random import *
import agents.random
from models.cnn import CNN
from models.autoencoder import Autoencoder

def AutoGreedyAgent(epochs=30, initial_epochs=None):
    '''Constructs agent with weighted autoencoder to predict sequence values that trains with each observation.
    Greedily selects sequences with best predicions.
    '''
    if initial_epochs is None:
        initial_epochs = epochs // 4

    class Agent(agents.random.RandomAgent(epochs, initial_epochs)):

        def __init__(self, *args):
            super().__init__(*args)
            self.model = Autoencoder(encoder=self.encode, 
                                        shape=self.shape, beta=1.)
            if len(self.prior):
                self.model.refit(*zip(*self.prior.items()), epochs=initial_epochs)
        
        def act(self, seqs):
            return list(zip(*sorted(zip(self.model.predict(seqs), seqs))[-self.batch:]))[1]

        def observe(self, data):
            super().observe(data)
            self.model.refit(*zip(*self.seen.items()), epochs=epochs) 
        
    return Agent


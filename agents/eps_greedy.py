import numpy as np
from random import *
import agents.random
from models.cnn import CNN

def EpsilonGreedyAgent(epochs=30, initial_epochs=None, eps=0.1):
    '''Constructs agent with CNN to predict sequence values that trains with each observation.
    Greedily selects sequences with best predicions. Act randomly with probability eps.
    '''
    if initial_epochs is None:
        initial_epochs = epochs // 4

    class Agent(agents.random.RandomAgent(epochs, initial_epochs)):

        def __init__(self, *args):
            super().__init__(*args)
            self.model = CNN(encoder=self.encode, shape=self.shape)
            if len(self.prior):
                self.model.fit(*zip(*self.prior.items()), epochs=initial_epochs, 
                                minibatch=100)
        
        def act(self, seqs):
            shuffle(seqs)
            selected = int((1 - eps) * self.batch)
            rand = seqs[:self.batch - selected]
            seqs = seqs[self.batch - selected:]
            return [*rand, *list(zip(*sorted(zip(self.model.predict(seqs), seqs))[-selected:]))[1]]

        def observe(self, data):
            super().observe(data)
            self.model.fit(*zip(*self.seen.items()), epochs=epochs, 
                            minibatch=min(len(self.seen), 100))
        
    return Agent


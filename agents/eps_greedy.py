import numpy as np
from random import *
import agents.random
from models.auto_cnn import CNN


def EpsilonGreedyAgent(epochs=30, eps=0.1):
    '''Constructs agent with CNN to predict sequence values that trains with each observation.
    Greedily selects sequences with best predicions. Act randomly with probability eps.
    '''

    class Agent(agents.random.RandomAgent(epochs)):

        def __init__(self, *args):
            super().__init__(*args)
            self.model = CNN(encoder=self.encode, shape=self.shape)
        
        def act(self, seqs):
            shuffle(seqs)
            selected = int((1 - eps) * self.batch)
            rand = seqs[:self.batch - selected]
            seqs = seqs[self.batch - selected:]
            return [*rand, *list(zip(*sorted(zip(self.model.predict(seqs), seqs))[-selected:]))[1]]

        def observe(self, data):
            super().observe(data)
            self.model.fit(*zip(*self.seen.items()), epochs=epochs)
        
    return Agent

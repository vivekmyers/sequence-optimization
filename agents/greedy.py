import numpy as np
from random import *
import agents.random
from models.auto_cnn import CNN


def GreedyAgent(epochs=30):
    '''Constructs agent with CNN to predict sequence values that trains with each observation.
    Greedily selects sequences with best predicions.
    '''

    class Agent(agents.random.RandomAgent(epochs)):

        def __init__(self, *args):
            super().__init__(*args)
            self.model = CNN(encoder=self.encode, shape=self.shape)
        
        def act(self, seqs):
            return list(zip(*sorted(zip(self.model.predict(seqs), seqs))[-self.batch:]))[1]

        def observe(self, data):
            super().observe(data)
            self.model.fit(*zip(*self.seen.items()), epochs=epochs)
        
    return Agent

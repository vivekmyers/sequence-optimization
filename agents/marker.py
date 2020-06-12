import numpy as np
from random import *
import agents.random
from models.marker import Marker
from models.auto_cnn import CNN
import utils.mcmc


def MarkerAgent(epochs=30, initial_epochs=None, dim=5, k=1., prior=(0.5, 10, 1, 1)):
    '''Constructs agent that marks sequences with embedding penalizing nearby mark sequences, then
    uses Thompson sampling to select between the clusters around each mark sequence in batches.
    dim: embedding shape
    beta: relative weight of sequence score in generating embedding
    k: number of clusters, or "affinity" for dynamic number
    prior: (mu0, n0, alpha, beta) prior over gamma and gaussian bucket score distributions
    '''
    if initial_epochs is None:
        initial_epochs = epochs

    class Agent(agents.random.RandomAgent(epochs, initial_epochs)):

        def __init__(self, *args):
            super().__init__(*args)
            self.model = Marker(init=self.prior, epochs=initial_epochs, encoder=self.encode, dim=dim, 
                                shape=self.shape, k=k, prior=prior)
        
        def act(self, seqs):
            return self.model.sample(seqs, self.batch)

        def observe(self, data):
            super().observe(data)
            self.model.fit(*zip(*data.items()), epochs=epochs)
        
    return Agent

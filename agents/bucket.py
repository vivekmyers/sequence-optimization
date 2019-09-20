import numpy as np
from random import *
import agents.random
from models.bucket import Bucketer
from models.auto_cnn import CNN
import utils.mcmc


def BucketAgent(epochs=30, initial_epochs=None, dim=5, k=1., prior=(0.5, 10, 1, 1)):
    '''Constructs agent that buckets sequences with autoencoder embedding, then
    uses Thompson sampling to select between buckets in batches.
    dim: embedding shape
    beta: relative weight of sequence score in generating embedding.
    k: number of clusters.
    prior: (mu0, n0, alpha, beta) prior over gamma and gaussian bucket score distributions.
    '''
    if initial_epochs is None:
        initial_epochs = epochs // 4

    class Agent(agents.random.RandomAgent(epochs, initial_epochs)):

        def __init__(self, *args):
            super().__init__(*args)
            self.model = Bucketer(encoder=self.encode, dim=dim, shape=self.shape, k=k, prior=prior)
            if len(self.prior):
                self.model.embed.fit(*zip(*self.prior.items()), epochs=initial_epochs)
        
        def act(self, seqs):
            return self.model.sample(seqs, self.batch)

        def observe(self, data):
            super().observe(data)
            self.model.fit(*zip(*self.seen.items()), epochs=epochs)
        
    return Agent

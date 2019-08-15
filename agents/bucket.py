import numpy as np
from random import *
import agents.random
from models.bucket import Bucketer
from models.cnn import CNN
import utils.mcmc

def BucketAgent(epochs=30, initial_epochs=None, dim=4, beta=0.5, k=1.):
    '''Constructs agent that autoencoder bucketing to thompson sample for sequences.
    dim: embedding shape
    beta: relative weight of sequence score in generating embedding.
    k: number of clusters.
    '''
    if initial_epochs is None:
        initial_epochs = epochs // 4

    class Agent(agents.random.RandomAgent(epochs, initial_epochs)):

        def __init__(self, *args):
            super().__init__(*args)
            self.model = Bucketer(encoder=self.encode, dim=dim, shape=self.shape, 
                                            beta=beta, k=k)
            if len(self.prior):
                self.model.embed.refit(*zip(*self.prior.items()), epochs=initial_epochs)
        
        def act(self, seqs):
            return self.model.sample(seqs, self.batch)
            

        def observe(self, data):
            super().observe(data)
            self.model.fit(*zip(*self.seen.items()), epochs=epochs)
        
    return Agent


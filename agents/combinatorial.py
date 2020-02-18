import numpy as np
from random import *
import agents.random
from models.combinator import Combinator
from models.auto_cnn import CNN
import utils.mcmc


def CombinatorialAgent(epochs=30, dim=5, k=1., prior=(0., 10, 1, 1), eps=0., rho=1.0):
    '''Constructs agent that buckets sequences with autoencoder embedding, then
    uses MCMC to approximate Thompson sampling over all possible distributions 
    of buckets to sample to maximize a metric which evaluates a portion of
    the agent's top selections proportional to rho. Each time a bucket is chosen by
    the Thompson sampling step, a sequence is e-greedily selected from it.
    dim: embedding shape
    beta: relative weight of sequence score in generating embedding
    k: number of clusters, or "affinity" for dynamic number
    prior: (mu0, n0, alpha, beta) prior over gamma and gaussian bucket score distributions
    eps: e-greedy epsilon parameter for greedy maximization step
    rho: top portion of batch on which to maximize score (should correspond to metric parameter)
    '''

    class Agent(agents.random.RandomAgent(epochs)):

        def __init__(self, *args):
            super().__init__(*args)
            self.model = Combinator(encoder=self.encode, dim=dim, shape=self.shape, k=k, prior=prior, eps=eps, rho=rho)
        
        def act(self, seqs):
            return self.model.sample(seqs, self.batch)

        def observe(self, data):
            super().observe(data)
            self.model.fit(*zip(*self.seen.items()), epochs=epochs)
        
    return Agent

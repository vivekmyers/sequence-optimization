import numpy as np
from random import *
import agents.random
from models.spgp import SparseGaussianProcess
from models.auto_cnn import CNN
import utils.mcmc


def SmartSparseGaussianAgent(epochs=30, dim=5, beta=0.02, k=1., M=1000):
    '''Constructs agent that uses batch version of GP-UCB algorithm to sample
    sequences with a deep kernel sparse gaussian process regression. Uses
    autoencoder for mu value prediction.
    dim: embedding dimension.
    beta: relative weight of sequence score in generating embedding.
    k: scaling of batch by which to oversample, and then find representative
        maximally-separated subset with mcmc.
    '''

    class Agent(agents.random.RandomAgent(epochs)):

        def __init__(self, *args):
            super().__init__(*args)
            self.model = SparseGaussianProcess(encoder=self.encode, dim=dim, shape=self.shape, 
                                                beta=beta, M=M)
        
        def act(self, seqs):
            choices = []
            t = 1 + len(self.seen) // self.batch
            D = len(seqs) + len(self.seen)
            beta = lambda t: 2 * np.log(D * t ** 2 * np.pi ** 2 / 3)
            _, sigma = self.model.interpolate(seqs)
            mu = self.model.embed.predict(seqs)
            seqs = np.array(seqs)
            ucb = mu + 2 * np.sqrt(beta(t + 1)) * sigma
            selected = np.argsort(ucb)[-int(k * self.batch):]
            if k != 1.:
                idx = utils.mcmc.mcmc(self.batch, 
                            self.model.embed(seqs[selected]),
                            iters=1000)
                selected = selected[idx]
            return seqs[selected]

        def observe(self, data):
            super().observe(data)
            self.model.fit(*zip(*self.seen.items()), epochs=epochs) 
        
    return Agent

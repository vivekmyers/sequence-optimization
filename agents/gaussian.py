import numpy as np
from random import *
import agents.random
from models.gp import GaussianProcess
from models.auto_cnn import CNN
import utils.mcmc


def GaussianAgent(epochs=30, initial_epochs=None, dim=5, tau=0.02, k=1., beta=1.):
    '''Constructs agent that uses batch version of GP-UCB algorithm to sample
    sequences with a deep kernel gaussian process regression.
    dim: embedding dimension.
    tau: kernel covariance parameter.
    beta: squared scaling of uncertainty for ucb.
    k: scaling of batch by which to oversample, and then find representative
        maximally-separated subset with mcmc.
    '''
    if initial_epochs is None:
        initial_epochs = epochs // 4

    class Agent(agents.random.RandomAgent(epochs, initial_epochs)):

        def __init__(self, *args):
            super().__init__(*args)
            self.k = k
            self.model = GaussianProcess(encoder=self.encode, dim=dim, shape=self.shape, tau=tau)
            self.beta = beta
            
            if len(self.prior):
                self.model.embed.fit(*zip(*self.prior.items()), epochs=initial_epochs)
        
        def act(self, seqs):
            t = 1 + len(self.seen) // self.batch
            D = len(seqs) + len(self.seen)
            beta = lambda t: beta * 2 * np.log(D * t ** 2 * np.pi ** 2 / 3)
            mu = self.model.interpolate(seqs)
            sigma = self.model.uncertainty(seqs)
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

def FixedGaussianAgent(*args, mb=10, **kwargs):
    '''Gaussian Agent that uses fixed scaling on uncertainty 
    (ucb = mu + sqrt(beta) * sigma) for beta fixed in batches of size mb.
    '''

    class Agent(GaussianAgent(*args, **kwargs)):

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def act(self, seqs):
            seqs = np.array(seqs)
            choices = []
            while len(choices) < self.batch:
                mu = self.model.interpolate(seqs)
                sigma = self.model.uncertainty(seqs, choices)
                ucb = mu + np.sqrt(self.beta) * sigma
                selected = np.argsort(ucb)[-int(self.k * mb):]
                if self.k != 1.:
                    idx = utils.mcmc.mcmc(mb, 
                                self.model.embed(seqs[selected]),
                                iters=1000)
                    selected = selected[idx]
                choices += list(seqs[selected])
                seqs = np.delete(seqs, selected)
            return choices[:self.batch]

    return Agent

def FeatureGaussianAgent(*args, **kwargs):

    class Agent(GaussianAgent(*args, **kwargs)):

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.model = FeautureGaussianProcess(encoder=self.encode, dim=dim, shape=self.shape,
                                            tau=tau, beta=beta, eps=0.01)
            if len(self.prior):
                self.model.embed.fit(*zip(*self.prior.items()), epochs=initial_epochs)

    return Agent

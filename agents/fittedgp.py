import numpy as np
from random import *
import agents.random
from models.exactgp import FittedGP
from models.featurizer import Featurizer
import utils.mcmc


def FittedGaussianAgent(epochs=30, initial_epochs=None, dim=5, beta=1.):
    '''Constructs agent that uses batch version of GP-UCB algorithm to sample
    sequences with a fitted GPyTorch regression.
    dim: embedding dimension.
    beta: squared scaling of uncertainty for ucb.
    mb: actions selected before refitting GP.
    '''
    if initial_epochs is None:
        initial_epochs = epochs // 4

    class Agent(agents.random.RandomAgent(epochs, initial_epochs)):

        def __init__(self, *args):
            super().__init__(*args)
            self.embed = Featurizer(self.encode, dim=dim, alpha=5e-4, shape=self.shape, lam=0., minibatch=100)
            self.beta = beta
            if len(self.prior):
                self.embed.fit(*zip(*self.prior.items()), epochs=initial_epochs)
        
        def act(self, seqs):
            if not self.seen.items():
                return sample(seqs, self.batch)
            seqs = np.array(seqs)
            X, Y = map(np.array, zip(*self.seen.items()))
            choices = []
            model = FittedGP(self.embed(X), Y)
            model.fit(epochs=epochs)
            mu, sigma = model.predict(self.embed(seqs))
            ucb = mu + np.sqrt(self.beta) * sigma
            selected = np.argsort(ucb)[-self.batch:]
            choices = list(seqs[selected])
            seqs = np.delete(seqs, selected)
            return choices

        def observe(self, data):
            super().observe(data)
            self.embed.fit(*zip(*self.seen.items()), epochs=epochs)
        
    return Agent


import numpy as np
from random import *
import agents.base
from models.gp import GaussianProcess

def GaussianAgent(epochs=200, initial_epochs=None, dim=5, tau=0.01):
    '''Constructs agent that uses batch version of GP-UCB algorithm to sample
    sequences with a deep kernel gaussian process regression.
    dim: embedding dimension
    tau: kernel covariance parameter
    '''
    if initial_epochs is None:
        initial_epochs = 2 * epochs

    class Agent(agents.base.BaseAgent):

        def __init__(self, *args):
            super().__init__(*args)
            self.model = GaussianProcess(encoder=self.encode, dim=dim, shape=self.shape, tau=tau)
            self.model.fit(*zip(*self.seen.items()), epochs=initial_epochs, 
                                minibatch=min(len(self.seen), 100))
        
        def act(self, seqs):
            prior = {}
            choices = []
            t = len(self.seen) // self.batch
            D = len(seqs) + len(self.seen)
            beta = lambda t: 2 * np.log(D * t ** 2 * np.pi ** 2 / 3)
            mu = self.model.interpolate(seqs, prior)
            sigma = self.model.uncertainty(seqs, prior)
            yt = (mu - np.sqrt(beta(t)) * sigma).max()
            seqs = np.array(seqs)
            x0 = seqs[np.argmax(mu + np.sqrt(beta(t)) * sigma)]
            Rt = seqs[mu + 2 * np.sqrt(beta(t + 1)) * sigma >= yt]
            prior[x0] = None
            choices = [x0]
            for i in range(self.batch - 1):
                sigma = self.model.uncertainty(Rt, prior)
                xk = Rt[np.argmax(sigma)]
                prior[xk] = None
                choices.append(xk)
            return choices

        def observe(self, data):
            super().observe(data)
            self.model.fit(*zip(*self.seen.items()), epochs=epochs, 
                                minibatch=min(len(self.seen), 100))
        
        def predict(self, seqs):
            return self.model.interpolate(seqs, {})

    return Agent


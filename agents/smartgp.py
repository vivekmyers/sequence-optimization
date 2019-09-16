import numpy as np
from random import *
import agents.base
from models.gp import GaussianProcess
from models.auto_cnn import CNN
from agents.gaussian import GaussianAgent


def SmartGaussianAgent(epochs=30, initial_epochs=None, dim=5, tau=0.01, beta=0.02):
    '''Constructs agent that uses batch version of GP-UCB algorithm to sample
    sequences with a deep kernel gaussian process regression.
    dim: embedding dimension. Uses autoencoder predictions instead of gaussian
    regression for computing predicted mu values.
    tau: kernel covariance parameter.
    beta: relative weight of sequence score in generating embedding.
    '''
    if initial_epochs is None:
        initial_epochs = epochs // 4

    class Agent(agents.gaussian.GaussianAgent(epochs, initial_epochs, dim, tau, beta)):

        def act(self, seqs):
            prior = {}
            choices = []
            t = 1 + len(self.seen) // self.batch
            D = len(seqs) + len(self.seen)
            beta = lambda t: 2 * np.log(D * t ** 2 * np.pi ** 2 / 3)
            mu = self.model.embed.predict(seqs)
            sigma = self.model.uncertainty(seqs, prior)
            seqs = np.array(seqs)
            ucb = mu + 2 * np.sqrt(beta(t + 1)) * sigma
            selected = np.argsort(ucb)[-self.batch:]
            return seqs[selected]

    return Agent

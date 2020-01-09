import numpy as np
from random import *
import agents.random
from models.exactgp import FittedGP
from models.featurizer import Featurizer
import utils.mcmc
import torch
from torch.distributions.multivariate_normal import MultivariateNormal


def FittedGaussianAgent(epochs=30, initial_epochs=None, dim=5, beta=1., mb=10):
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
            mu = None
            while len(choices) < self.batch:
                model = FittedGP(self.embed(X), Y)
                model.fit(epochs=epochs)
                mu_, sigma = model.predict(self.embed(seqs))
                if mu is None:
                    mu = mu_
                ucb = mu + np.sqrt(self.beta) * sigma
                selected = np.argsort(ucb)[-mb:]
                choices += list(seqs[selected])
                X = np.concatenate((X, seqs[selected]))
                Y = np.concatenate((Y, mu[selected]))
                seqs = np.delete(seqs, selected)
            return choices[:self.batch]

        def observe(self, data):
            super().observe(data)
            self.embed.fit(*zip(*self.seen.items()), epochs=epochs)
        
    return Agent


def ThompsonGPAgent(epochs=30, initial_epochs=None, dim=5):
    '''Agent using batch GP Thompson sampling.'''
    if initial_epochs is None:
        initial_epochs = epochs // 4

    class Agent(agents.random.RandomAgent(epochs, initial_epochs)):

        def __init__(self, *args):
            super().__init__(*args)
            self.embed = Featurizer(self.encode, dim=dim, alpha=5e-4, shape=self.shape, lam=0., minibatch=100)
            if len(self.prior):
                self.embed.fit(*zip(*self.prior.items()), epochs=initial_epochs)
        
        def act(self, seqs):
            if not self.seen.items():
                return sample(seqs, self.batch)
            seqs = np.array(seqs)
            X, Y = map(np.array, zip(*self.seen.items()))
            model = FittedGP(self.embed(X), Y)
            model.fit(epochs=epochs)
            mu, cov = model.predict_(self.embed(seqs))
            mvn = MultivariateNormal(torch.tensor(mu), covariance_matrix=torch.tensor(cov) + 1e-4 * torch.eye(cov.shape[0])) 
            mask = np.array([False for _ in seqs])
            choices = []
            for i in range(self.batch):
                samp = mvn.sample().data.numpy()
                low = samp.min()
                samp[mask] = low
                idx = np.argmax(samp)
                mask[idx] = True
                choices.append(seqs[idx])
            return choices

        def observe(self, data):
            super().observe(data)
            self.embed.fit(*zip(*self.seen.items()), epochs=epochs)
        
    return Agent


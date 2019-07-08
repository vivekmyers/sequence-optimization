import numpy as np
from random import *
import os, sys
import torch
from torch import nn
import torch.functional as F
from models.embed import Embedding
from scipy.spatial.distance import pdist, cdist, squareform

class GaussianProcess:
    '''Fits gaussian process model to sequence data using a deep kernel function.'''

    def fit(self, seqs, scores, epochs, minibatch):
        self.X = seqs[:]
        self.Y = scores[:]
        self.embed.refit(self.X, self.Y, epochs, minibatch)

    def interpolate(self, x, prior):
        '''Given observed points in (self.X, self.Y) and points (X, Y) in prior.items(),
        fit gaussian process regression and return means predicted for each
        provided point in x.
        '''
        X = [*self.X]
        Y = [*self.Y]
        for a, b in prior.items():
            X.append(a)
            Y.append(b)
        X, Y, x = map(np.array, [X, Y, x])
        X, x = map(self.embed, [X, x])
        K_XX = self.sigma * np.exp(-1 / (2 * self.tau ** 2) * squareform(pdist(X)) ** 2)
        K_star = self.sigma * np.exp(-1 / (2 * self.tau ** 2) * cdist(x, X).reshape([len(x), len(X)]) ** 2)
        mu = self.mu + np.squeeze(K_star @ np.linalg.inv(K_XX) @ (Y[:, None] - self.mu))
        return mu

    def uncertainty(self, x, prior):
        '''Given observed points in self.X and keys X in prior.keys(), fits gaussian
        process regression and returns predicted sigmas for each point in x.
        '''
        X = [*self.X] + [*prior.keys()]
        X, x = map(self.embed, map(np.array, [X, x]))
        K_XX = self.sigma * np.exp(-1 / (2 * self.tau ** 2) * squareform(pdist(X)) ** 2)
        K_star = self.sigma * np.exp(-1 / (2 * self.tau ** 2) * cdist(x, X).reshape([len(x), len(X)]) ** 2)
        sigma = self.sigma - K_star @ np.linalg.inv(K_XX) @ K_star.T
        return np.diagonal(sigma)

    def __init__(self, encoder, dim, shape=(), alpha=1e-4, lam=1e-3, mu=0.5, sigma=0.5, tau=1):
        '''encoder: convert sequences to one-hot arrays.
        alpha: embedding learning rate.
        shape: sequence shape (len, channels)
        lam: l2 regularization constant
        mu: GP prior mean
        sigma: GP prior standard deviation
        '''
        super().__init__()
        self.X, self.Y = (), ()
        self.embed = Embedding(encoder, dim, alpha=alpha, shape=shape, lam=lam)
        self.mu = mu
        self.sigma = sigma
        self.tau = tau


import numpy as np
from random import *
import os, sys
import torch
from torch import nn
import torch.functional as F
from models.autoencoder import Autoencoder
from scipy.spatial.distance import pdist, cdist, squareform
from models.embed import *
import utils.model

class GaussianProcess:
    '''Fits gaussian process model to sequence data using a deep kernel function.'''

    def fit(self, seqs, scores, epochs):
        self.X = seqs[:]
        self.Y = scores[:]
        self.embed.refit(self.X, self.Y, epochs)

    @utils.model.batch
    def interpolate(self, x):
        '''Given observed points in (self.X, self.Y),
        fit gaussian process regression and return means predicted for each
        provided point in x.
        '''
        X = [*self.X]
        Y = [*self.Y]
        if len(X) == 0 or len(x) == 0:
            return np.full([len(x)], self.mu)
        X, Y, x = map(np.array, [X, Y, x])
        X, x = map(self.embed, [X, x])
        T = lambda t: torch.tensor(t).to(self.embed.device).double()
        K_XX = self.sigma ** 2 * torch.exp(-1 / (2 * self.tau ** 2) * T(squareform(pdist(X))) ** 2)
        K_star = self.sigma ** 2 * torch.exp(-1 / (2 * self.tau ** 2) * T(cdist(x, X)).view([len(x), len(X)]) ** 2)
        mu = self.mu + torch.squeeze(K_star @ torch.inverse(K_XX + \
            torch.eye(len(X)).to(self.embed.device).double() * self.eps) @ (T(Y)[:, None] - self.mu))
        return mu.cpu().numpy()

    @utils.model.batch
    def uncertainty(self, x):
        '''Given observed points in self.X, fits gaussian
        process regression and returns predicted sigmas for each point in x.
        '''
        X = [*self.X]
        if len(X) == 0 or len(x) == 0:
            return np.full([len(x)], self.sigma)
        X, x = map(self.embed, map(np.array, [X, x]))
        T = lambda t: torch.tensor(t).to(self.embed.device).double()
        K_XX = self.sigma ** 2 * torch.exp(-1 / (2 * self.tau ** 2) * T(squareform(pdist(X))) ** 2)
        K_star = self.sigma ** 2 * torch.exp(-1 / (2 * self.tau ** 2) * T(cdist(x, X)).view([len(x), len(X)]) ** 2)
        sigma = torch.sqrt(self.sigma ** 2 + self.eps - K_star @ torch.inverse(K_XX + \
                torch.eye(len(X)).to(self.embed.device).double() * self.eps) @ K_star.permute(1, 0))
        return np.diagonal(sigma.cpu().numpy())

    def __init__(self, encoder, dim, shape, beta=0., alpha=5e-4, 
                    lam=1e-5, mu=0.5, sigma=0.5, tau=1, eps=0.01, minibatch=100, gpbatch=5000):
        '''encoder: convert sequences to one-hot arrays.
        alpha: embedding learning rate.
        shape: sequence shape (len, channels)
        beta:embedding score weighting
        dim: embedding dimensionality
        lam: l2 regularization constant
        mu: GP prior mean
        sigma: GP prior standard deviation
        tau: kernel covariance parameter
        eps: noise
        '''
        super().__init__()
        self.X, self.Y = (), ()
        self.minibatch = gpbatch
        self.embed = Autoencoder(encoder, dim=dim, alpha=alpha, shape=shape, 
                                    lam=lam, beta=beta, minibatch=minibatch)
        self.mu = mu
        self.sigma = sigma
        self.tau = tau
        self.eps = eps

class FeautureGaussianProcess(GaussianProcess):

    def __init__(self, encoder, dim, shape, beta=0., alpha=5e-4,
                    lam=1e-3, mu=0.5, sigma=0.5, tau=1, eps=1e-4, minibatch=100, gpbatch=5000):
        self.X, self.Y = (), ()
        self.minibatch = gpbatch
        self.embed = DeepFeatureEmbedding(encoder, dim=dim, alpha=alpha, 
                        shape=shape, lam=lam, minibatch=minibatch)
        self.mu = mu
        self.sigma = sigma
        self.tau = tau
        self.eps = eps


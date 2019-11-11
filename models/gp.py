import numpy as np
from random import *
import os, sys
import torch
from torch import nn
import torch.functional as F
from models.featurizer import Featurizer
from scipy.spatial.distance import pdist, cdist, squareform
from models.embed import *
import utils.model


class GaussianProcess:
    '''Fits gaussian process model to sequence data using a deep kernel function.'''

    def fit(self, seqs, scores, epochs):
        self.X = seqs[:]
        self.Y = scores[:]
        self.embed.fit(self.X, self.Y, epochs)

    def mll(self, epochs=50):
        '''Fit RBF kernel parameters by minimizing -mll.'''
        T = lambda t: torch.tensor(t).to(self.embed.device).double()
        X = T(self.embed(self.X))
        n = len(self.Y)
        Y = T(self.Y).view(n, 1)
        for ep in range(epochs):
            K_XX = self.sigma ** 2 * torch.exp(-1 / (2 * self.tau ** 2) * T(squareform(pdist(X.cpu()))) ** 2)
            noise = torch.eye(n).to(self.embed.device).double() * self.eps
            mll = -0.5 * (Y - self.mu).t() @ torch.inverse(K_XX + noise) @ (Y - self.mu) - 0.5 * torch.det(K_XX + noise)
            self.opt.zero_grad()
            (-mll).backward()
            self.opt.step()

    @utils.model.batch
    def interpolate(self, x):
        '''Given observed points in (self.X, self.Y),
        fit gaussian process regression and return means predicted for each
        provided point in x.
        '''
        X = [*self.X]
        Y = [*self.Y]
        if len(X) == 0 or len(x) == 0:
            return np.full([len(x)], self.mu.cpu().item())
        X, Y, x = map(np.array, [X, Y, x])
        X, x = map(self.embed, [X, x])
        T = lambda t: torch.tensor(t).to(self.embed.device).double()
        K_XX = self.sigma ** 2 * torch.exp(-1 / (2 * self.tau ** 2) * T(squareform(pdist(X))) ** 2)
        K_star = self.sigma ** 2 * torch.exp(-1 / (2 * self.tau ** 2) * T(cdist(x, X)).view([len(x), len(X)]) ** 2)
        mu = self.mu + torch.squeeze(K_star @ torch.inverse(K_XX + \
            torch.eye(len(X)).to(self.embed.device).double() * self.eps) @ (T(Y)[:, None] - self.mu))
        return mu.detach().cpu().numpy()

    @utils.model.batch
    def uncertainty(self, x, prior=[]):
        '''Given observed points in self.X, fits gaussian
        process regression and returns predicted sigmas for each point in x.
        '''
        X = [*self.X, *prior]
        if len(X) == 0 or len(x) == 0:
            return np.full([len(x)], self.sigma.cpu().item())
        X, x = map(self.embed, map(np.array, [X, x]))
        T = lambda t: torch.tensor(t).to(self.embed.device).double()
        K_XX = self.sigma ** 2 * torch.exp(-1 / (2 * self.tau ** 2) * T(squareform(pdist(X))) ** 2)
        K_star = self.sigma ** 2 * torch.exp(-1 / (2 * self.tau ** 2) * T(cdist(x, X)).view([len(x), len(X)]) ** 2)
        sigma = torch.sqrt(self.sigma ** 2 + self.eps - K_star @ torch.inverse(K_XX + \
                torch.eye(len(X)).to(self.embed.device).double() * self.eps) @ K_star.permute(1, 0))
        return np.diagonal(sigma.detach().cpu().numpy())

    def __init__(self, encoder, dim, shape, alpha=5e-4, beta=0.05,
                    lam=0, mu=0.5, sigma=0.5, eps=1e-4, tau=1., minibatch=100, gpbatch=5000):
        '''encoder: convert sequences to one-hot arrays.
        alpha: embedding learning rate.
        shape: sequence shape (len, channels)
        dim: embedding dimensionality
        lam: l2 regularization constant
        mu: GP prior mean
        sigma: GP prior standard deviation
        tau: kernel covariance parameter
        beta: GP hyperparameter fitting rate
        eps: noise
        '''
        super().__init__()
        self.X, self.Y = (), ()
        self.minibatch = gpbatch
        self.embed = Featurizer(encoder, dim=dim, alpha=alpha, shape=shape, 
                                    lam=lam, minibatch=minibatch)
        self.mu = mu
        self.sigma = sigma
        self.tau = torch.tensor(tau, requires_grad=True, device=self.embed.device, dtype=torch.double)
        self.mu = torch.tensor(mu, requires_grad=True, device=self.embed.device, dtype=torch.double)
        self.sigma = torch.tensor(sigma, requires_grad=True, device=self.embed.device, dtype=torch.double)
        self.eps = torch.tensor(eps, requires_grad=True, device=self.embed.device, dtype=torch.double)
        self.opt = torch.optim.Adam([self.tau, self.mu, self.sigma, self.eps], lr=beta)



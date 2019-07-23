import numpy as np
from random import *
import os, sys
import torch
from torch import nn
import torch.functional as F
from torch.distributions import Normal
from models.autoencoder import Autoencoder
from scipy.spatial.distance import pdist, cdist, squareform

class SparseGaussianProcess:
    '''Fits gaussian process model using induced points.'''

    def fit(self, seqs, scores, epochs, minibatch):
        self.X = seqs[:]
        self.Y = scores[:]
        self.embed.refit(self.X, self.Y, epochs, minibatch)

    def _induce(self, X, M, Y, X_pred):
        n = len(X)
        T = lambda t: torch.tensor(t, requires_grad=True).to(self.embed.device).float()
        X, Y = map(T, [X, Y])
        X_bar = torch.empty(M, X.shape[1])
        for i, t in enumerate(sample(X, M)):
            X_bar.data[i, :] = t
        c, b, sig = T(1.), T([1.] * dim), T(1.)
        K = lambda x, y: c * torch.exp(-1 / 2 * torch.sum(b * (x - y) ** 2)

        K_MN = T([[K(a, b) for a in X] for b in X_bar])
        K_NM = K_MN.permute(0, 1)
        K_M = T([[K(a, b) for a in X_bar] for b in X_bar])
        lam = torch.diag(T([K(x, x) - torch.squeeze(K_NM[i].view(1, n) @ \
                    torch.inverse(K_M) @ K_NM[i].view(n, 1)) for x, i in enumerate(X)]))
        I = torch.eye(n) * sig
        sigma = K_NM @ torch.inverse(K_M) @ K_MN + lam + I
        prior = T([self.mu] * n)
        loss = -Normal(prior, sigma).log_prob(Y)
        opt = torch.optim.Adam([X_bar, c, b, sig], lr=self.alpha)

        for itr in range(self.itr):
            opt.zero_grad()
            loss.backward(retain_graph=itr < self.itr - 1)
            opt.step()

        Q_M = K_M + K_MN @ torch.inverse(lam + I) @ K_NM
        K_star = T([[K(x, x_bar) for x_bar in X_bar] for x in X_pred])
        mu = torch.squeeze(self.mu + K_star @ torch.inverse(Q_M) @ K_MN @ torch.inverse(lam + I) @ (Y[:, None] - self.mu), dim=1)
        sig_sq = T([K(x, x) for x in X_pred]) - K_star @ (torch.inverse(K_M) - torch.inverse(Q_M)) @ K_star.permute(1, 0) + sig
        return mu, sig_sq

    def interpolate(self, x, prior={}):
        '''Given observed points in (self.X, self.Y) and points (X, Y) in prior.items(),
        fit gaussian process regression and return mus, sigmas predicted for each
        provided point in x.
        '''
        X = [*self.X]
        Y = [*self.Y]
        for a, b in prior.items():
            X.append(a)
            Y.append(b)
        if len(X) == 0 or len(x) == 0:
            return np.full([len(x)], self.mu)
        mu, sig_sq = self._induce(X, min(self.M, len(X)), Y, x)
        return mu.detach().cpu().numpy(), torch.sqrt(sig_sq).detach().cpu().numpy()


    def __init__(self, encoder, dim, shape, beta=0., alpha=5e-4, 
                    lam=1e-3, mu=0.5, itr=1000, M=100):
        '''encoder: convert sequences to one-hot arrays.
        alpha: embedding and induced point descent learning rate.
        shape: sequence shape (len, channels).
        beta: embedding score weighting.
        dim: embedding dimensionality.
        lam: l2 regularization constant.
        mu: GP prior mean.
        M: max number of induced points.
        itr: gradient ascent iterations for induced pseudo-inputs.
        '''
        super().__init__()
        self.X, self.Y = (), ()
        self.embed = Autoencoder(encoder, dim=dim, alpha=alpha, shape=shape, lam=lam, beta=beta)
        self.mu = mu
        self.sigma = sigma
        self.dim = dim
        self.tau = tau
        self.eps = eps
        self.alpha = alpha
        self.itr = 1000
        self.M = M


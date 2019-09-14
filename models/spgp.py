import numpy as np
from random import *
import os, sys
import torch
from torch import nn
import torch.functional as F
from torch.distributions import *
from models.autoencoder import Autoencoder
from scipy.spatial.distance import pdist, cdist, squareform
import utils.model

class SparseGaussianProcess:
    '''Fits gaussian process model using induced points.'''

    def fit(self, seqs, scores, epochs):
        self.X = seqs[:]
        self.Y = scores[:]
        self.embed.fit(self.X, self.Y, epochs)

    def _induce(self, X, M, Y, X_pred):
        n = len(X)
        T = lambda t: torch.tensor(t, requires_grad=True, device=self.embed.device, dtype=torch.double)
        X, Y, X_pred = map(T, [X, Y, X_pred])
        X_bar = torch.empty(M, X.shape[1], device=self.embed.device, requires_grad=True, dtype=torch.double)
        for i, t in enumerate(sample(list(X), M)):
            X_bar.data[i, :] = t
        c, b, sig = T(1.), T([1.] * X.size(1)), T(1.)
        K = lambda x, y: c * torch.exp(-1 / 2 * torch.sum(b * (x - y) ** 2))
        K_vec = lambda x, y: c * torch.exp(-1 / 2 * 
                    torch.sum(b[None, None, :] * (x[None, :, :] - y[:, None, :]) ** 2, 
                        dim=2)).transpose(0, 1)
        I = torch.eye(n).to(self.embed.device).double()
        opt = torch.optim.Adam([X_bar, c, b, sig], lr=self.zeta)

        for itr in range(self.itr):
            K_MN = K_vec(X_bar, X)
            K_NM = K_MN.transpose(0, 1)
            K_M = K_vec(X_bar, X_bar) + self.eps * torch.eye(M).to(self.embed.device).double()
            lam = torch.diag(torch.stack([K(x, x) for x in X]) - torch.diag(K_NM @ torch.inverse(K_M) @ K_MN))
            sigma = K_NM @ torch.inverse(K_M) @ K_MN + lam + I * sig.exp().add(1).log()
            prior = T([self.mu] * n)
            try:
                loss = -MultivariateNormal(prior, sigma).log_prob(Y).sum()
                opt.zero_grad()
                loss.backward()
                opt.step()
            except RuntimeError:
                break

        Q_M = K_M + K_MN @ torch.inverse(lam + I * sig.exp().add(1).log()) @ K_NM
        K_star = K_vec(X_pred, X_bar)
        mu = torch.squeeze(self.mu + K_star @ torch.inverse(Q_M) @ K_MN @ \
                torch.inverse(lam + I * sig.exp().add(1).log()) @ (Y[:, None] - self.mu), dim=1)
        sig_sq = T([K(x, x) for x in X_pred]) - K_star.view(K_star.size(0), 1, K_star.size(1)).bmm(((torch.inverse(K_M) - torch.inverse(Q_M)) \
                @ K_star.permute(1, 0)).transpose(0, 1).view(K_star.size(0), K_star.size(1), 1)).view(X_pred.size(0)) + sig.exp().add(1).log()
        return mu, sig_sq

    @utils.model.batch
    def interpolate(self, x):
        '''Given observed points in (self.X, self.Y),
        fit gaussian process regression and return mus, sigmas predicted for each
        provided point in x.
        '''
        X = [*self.X]
        Y = [*self.Y]
        if len(X) == 0 or len(x) == 0:
            return tuple(np.full([2, len(x)], self.mu))
        mu, sig_sq = self._induce(self.embed(X), min(self.M, len(X)), Y, self.embed(x))
        return mu.detach().cpu().numpy(), torch.sqrt(sig_sq).detach().cpu().numpy()


    def __init__(self, encoder, dim, shape, beta=0., alpha=5e-4, 
                    zeta=1e-2, lam=1e-6, mu=0.5, itr=200, M=1000, eps=1e-4, minibatch=100, gpbatch=2000):
        '''encoder: convert sequences to one-hot arrays.
        alpha: embedding learning rate.
        zeta: induced point ascent learning rate
        shape: sequence shape (len, channels).
        beta: embedding score weighting.
        dim: embedding dimensionality.
        lam: l2 regularization constant.
        mu: GP prior mean.
        M: max number of induced points.
        itr: gradient ascent iterations for induced pseudo-inputs.
        eps: numerical stability
        '''
        super().__init__()
        self.X, self.Y = (), ()
        self.minibatch = gpbatch
        self.embed = Autoencoder(encoder, dim=dim, alpha=alpha, shape=shape, 
                                    lam=lam, beta=beta, minibatch=minibatch)
        self.mu = mu
        self.dim = dim
        self.alpha = alpha
        self.itr = itr
        self.eps = eps
        self.M = M
        self.zeta = zeta


import numpy as np
from random import *
import os, sys
import torch
from torch import nn
import torch.functional as F
from models.autoencoder import Autoencoder
from torch.distributions import Normal
from scipy.spatial.distance import pdist, cdist, squareform
from sklearn.cluster import KMeans

class Bucketer:
    '''Buckets and samples from embedded sequences with TS.'''

    def fit(self, seqs, scores, epochs, minibatch):
        self.X = seqs[:]
        self.Y = scores[:]
        self.embed.refit(self.X, self.Y, epochs, minibatch)

    def sample(self, pts, n):
        pts = pts[:]
        em = list(self.embed(self.X)) if len(self.X) else []
        pts_em = list(self.embed(pts))
        model = KMeans(self.k).fit(em + pts_em)
        vals = [[] for i in range(self.k)]
        for idx, val in zip(model.predict(em) if len(em) else [], self.Y):
           vals[idx].append(val)
        mus = np.array([np.array(x).mean() if x else 0 for x in vals])
        sigmas = np.array([np.array(x).std() if x else 1 for x in vals])
        dist = [Normal(mu * self.sigma **  2 / (self.sigma ** 2 + sigma ** 2 / len(mus)) + \
                self.mu * sigma ** 2 / (sigma ** 2 / len(mus) + self.sigma ** 2), \
                1 / (1 / sigma ** 2 + 1 / self.sigma ** 2)) for mu, sigma in zip(mus, sigmas)]
        ret = []
        for i in range(n):
            sampled = np.argmax(np.array([d.sample().item() for d in dist]))
            clust = np.array(pts)[model.predict(pts_em) == sampled]
            if len(clust) == 0:
                ret.append(choice(pts))
            else:
                ret.append(clust[np.argmax(self.embed.predict(clust))])
            del pts_em[pts.index(ret[-1])]
            del pts[pts.index(ret[-1])]
        return ret


    def __init__(self, encoder, dim, shape, beta=0., alpha=5e-4, 
                    lam=1e-6, sigma=0.5, mu=0.5, tau=1, eps=0.01, k=100):
        '''encoder: convert sequences to one-hot arrays.
        alpha: embedding learning rate.
        shape: sequence shape (len, channels)
        beta: embedding score weighting
        dim: embedding dimensionality
        lam: l2 regularization constant
        mu: prior mean
        sigma: prior standard deviation
        k: cluster count
        '''
        super().__init__()
        self.X, self.Y = (), ()
        self.embed = Autoencoder(encoder, dim=dim, alpha=alpha, shape=shape, lam=lam, beta=beta)
        self.mu = mu
        self.sigma = sigma
        self.tau = tau
        self.eps = eps
        self.k = k


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

    def fit(self, seqs, scores, epochs):
        self.X = seqs[:]
        self.Y = scores[:]
        self.embed.refit(self.X, self.Y, epochs)

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
        dist = [Normal(mu, 1 / (len(val) / sigma ** 2 + 1 / self.sigma ** 2)) for mu, sigma, val in zip(mus, sigmas, vals)]
        ret = []
        pred = model.predict(pts_em)
        scores = self.embed.predict(pts)
        for i in range(n):
            sampled = np.argmax(np.array([d.sample().item()
                        if i in pred else min(scores)
                        for i, d in enumerate(dist)]))
            clust = np.array(pts)[pred == sampled]
            clust_scores = np.array(scores)[pred == sampled]
            ret.append(clust[np.argmax(clust_scores)])
            del pts_em[pts.index(ret[-1])]
            pred = pred[np.arange(pred.shape[0]) != pts.index(ret[-1])]
            scores = scores[np.arange(scores.shape[0]) != pts.index(ret[-1])]
            del pts[pts.index(ret[-1])]
        return ret


    def __init__(self, encoder, dim, shape, beta=0.5, alpha=5e-4, 
                    sigma=0.5, mu=0.5, k=100, minibatch=100):
        '''encoder: convert sequences to one-hot arrays.
        alpha: embedding learning rate.
        shape: sequence shape (len, channels)
        beta: embedding score weighting
        dim: embedding dimensionality
        mu: prior mean
        sigma: prior standard deviation
        k: cluster count
        '''
        super().__init__()
        self.X, self.Y = (), ()
        self.embed = Autoencoder(encoder, dim=dim, alpha=alpha, 
                        shape=shape, beta=beta, minibatch=minibatch)
        self.mu = mu
        self.sigma = sigma
        self.k = k


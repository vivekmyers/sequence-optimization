import numpy as np
from random import *
import os, sys
import torch
from torch import nn
import torch.functional as F
from models.mark import MarkEmbedding
from sklearn.cluster import KMeans, AffinityPropagation
from sklearn.metrics import silhouette_score


class Marker:
    '''Selects mark sequences and samples from embedded sequences with Thompson sampling.'''

    def fit(self, seqs, scores, epochs):
        '''Fits model to observed labeled sequences. Should be
        called with all new labeled sequences seen so far at each
        time step.
        '''
        self.X = [*seqs, *self.X]
        self.Y = [*scores, *self.Y]
        closest = self._closest(seqs)
        for x, y, m in zip(seqs, scores, closest):
            self.seen[x] = y
            if self.seen[self.markers[m]] < y:
                self.markers[m] = x
        self.embed.fit(self.X, self.Y, epochs, self.markers)

    def _closest(self, X):
        return np.argmin(np.linalg.norm(self.embed(self.markers)[None, :, :] - self.embed(X)[:, None, :], axis=2), axis=1)

    def sample(self, pts, n):
        '''Thompson sample sequences.
        pts: sequences to sample from
        n: number of sequences to sample
        '''
        pts = pts[:]
        total = list(self.X) + list(pts)
        locations = {x: y for x, y in zip(total, self._closest(total))}
        buckets = [[] for _ in range(self.k)]
        unseen = [[] for _ in range(self.k)]
        
        for x in self.X:
            buckets[locations[x]].append(self.seen[x])

        for x in pts:
            unseen[locations[x]].append(x)

        buckets = list(map(np.array, buckets))
        unseen = list(map(np.array, unseen))


        mu0, n0, alpha, beta = self.prior # unpack prior parameters

        def conjugate(x):
            '''Returns conjugate distribution over mean mu and standard deviation of mu
            fixing sample from gamma distribution.
            '''
            mu0, n0, alpha, beta = self.prior # unpack prior parameters
            n = len(x)
            mu = x.mean() if n else mu0
            a = alpha + n / 2
            b0 = beta + 1 / 2 * ((x - mu) ** 2).sum()
            b1 = n * n0 * (mu - mu0) ** 2 / (2 * (n + n0))
            b = 1 / (b0 + b1)
            mu_exp = (n * mu + n0 * mu0) / (n + n0)
            mu_var_scale = 1 / (n + n0)

            def sample():
                tau = np.random.gamma(a, b)
                mu_var = mu_var_scale / tau
                return np.random.normal(mu_exp, np.sqrt(mu_var)), mu_var

            return sample

        # construct conjugate distributions for each bucket
        conj_dists = [conjugate(x) for x in buckets]
        dists = [lambda dist=dist: dist()[0] for dist in conj_dists]

        # select n sequences to return
        selections = []
        scores = [self.embed.predict(vals) for vals in unseen]
        valid = [i for i in range(self.k) if len(unseen[i])]
        
        for i in range(n):

            # 1. Thompson sample a bucket by sampling from each conjugate dist and taking max
            # 2. get the unlabeled sequences in it and their predictions
            samples = np.array([dist() if bucket_idx in valid else -np.inf
                                        for bucket_idx, dist in enumerate(dists)])
            sampled_bucket_idx = np.argmax(samples)
            sampled_pts_idx = np.argmax(scores[sampled_bucket_idx])
            selections.append(unseen[sampled_bucket_idx][sampled_pts_idx])
            unseen[sampled_bucket_idx] = np.delete(unseen[sampled_bucket_idx], sampled_pts_idx)
            scores[sampled_bucket_idx] = np.delete(scores[sampled_bucket_idx], sampled_pts_idx)
            if len(unseen[sampled_bucket_idx]) == 0:
                valid.remove(sampled_bucket_idx)


        return selections

    def __init__(self, init, epochs, encoder, dim, shape, alpha=5e-4,
                    prior=(0.5, 10, 1, 1), eps=0., rho=0., k=100, minibatch=100):
        '''init: initial data
        epochs: start training epochs
        encoder: convert sequences to one-hot arrays.
        alpha: embedding learning rate.
        shape: sequence shape (len, channels)
        dim: embedding dimensionality
        prior: (mu0, n0, alpha, beta) prior over gamma and gaussian bucket score distributions.
        eps: epsilon for greedy maximization step
        rho: portion of steps to use inverse gamma conjugate instead of normal gamma
        k: cluster count or method
        '''
        super().__init__()
        self.embed = MarkEmbedding(encoder=encoder, shape=shape, dim=dim, alpha=alpha, minibatch=minibatch)
        X, Y = map(np.array, zip(*[x for x in init.items()]))
        self.X = X[:]
        self.Y = Y[:]
        self.prior = prior
        self.eps = eps
        self.rho = rho
        self.k = k
        self.seen = {x: y for x, y in zip(X, Y)}

        def distfrom(x, markers):
            return min(np.abs(encoder(x) - encoder(m)).sum() for m in markers)

        # Compute initial marker sequences in the top 10%
        Y_cut = sorted(Y)[int(0.9*len(Y))]
        idx = np.argmax(Y)
        self.markers = [X[idx]]
        X = np.delete(X, idx)
        Y = np.delete(Y, idx)
        X_pos = X[Y > Y_cut]
        Y_pos = Y[Y > Y_cut]
        for i in range(1, k):
            idx = max(range(len(X_pos)), key=lambda x: distfrom(X_pos[x], self.markers))
            self.markers.append(X_pos[idx])
            X_pos = np.delete(X_pos, idx)
            Y_pos = np.delete(Y_pos, idx)
        
        self.embed.fit(X, Y, epochs, self.markers)



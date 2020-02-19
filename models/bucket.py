import numpy as np
from random import *
import os, sys
import torch
from torch import nn
import torch.functional as F
from models.featurizer import Featurizer
from sklearn.cluster import KMeans, AffinityPropagation
from sklearn.metrics import silhouette_score


class Bucketer:
    '''Buckets and samples from embedded sequences with Thompson sampling.'''

    def fit(self, seqs, scores, epochs):
        '''Fits model to observed labeled sequences. Should be
        called with all labeled sequences seen so far at each
        time step.
        '''
        self.X = seqs[:]
        self.Y = scores[:]
        self.embed.fit(self.X, self.Y, epochs)

    def sample(self, pts, n):
        '''Thompson sample sequences.
        pts: sequences to sample from
        n: number of sequences to sample
        '''
        pts = pts[:]
        seen_em = list(self.embed(self.X)) if len(self.X) else [] # embedding of seen sequences
        pts_em = list(self.embed(pts)) # embedding of unlabeled sequences

        # create buckets containing labels of seen sequences using k-means
        # of embeddings of both seen and unseen sequences
        if self.k == 'affinity':
            method = AffinityPropagation() 
        elif self.k == 'silhouette':
            scores = [silhouette_score(seen_em + pts_em, KMeans(n).fit_predict(seen_em + pts_em)) 
                            for n in range(2, 100)]
            method = KMeans(np.array(scores).argmax() + 1)
        else:
            method = KMeans(self.k)
        clustering = method.fit(seen_em + pts_em)
        k = 1 + np.max(clustering.predict(seen_em + pts_em))
        buckets = [[] for i in range(k)]
        for idx, val in zip(clustering.predict(seen_em) if len(seen_em) else [], self.Y):
           buckets[idx].append(val)
        buckets = list(map(np.array, buckets))

        mu0, n0, alpha, beta = self.prior # unpack prior parameters

        def conj_tau(x): 
            '''Returns a, b where the conjugate distribution of tau
            given x is Ga(a, b).
            '''
            if len(x) == 0: return alpha, 1 / beta
            a = alpha + len(x) / 2
            b0 = beta + 1 / 2 * ((x - x.mean()) ** 2).sum()
            b1 = len(x) * n0 * (x.mean() - mu0) ** 2 / (2 * (len(x) + n0))
            return a, 1 / (b0 + b1)

        def conj_mu(x, tau):
            '''Returns mu', sigma' where the conjugate distribution of
            mu given x, tau is N(mu', sigma').
            '''
            n = len(x)
            if n == 0: return mu0, np.sqrt(1 / (n0 * tau))
            mu = (n * x.mean() + n0 * mu0) / (n + n0)
            prec = (n + n0) * tau
            return mu, np.sqrt(1 / prec)

        # construct conjugate distributions for each bucket
        taus = [lambda x=x: np.random.gamma(*conj_tau(x)) for x in buckets]
        mu_dists = [lambda x=x, tau=tau: np.random.normal(*conj_mu(x, tau()))
                            for x, tau in zip(buckets, taus)]
        sigma_dists = [lambda x=x, tau=tau: conj_mu(x, tau())[1]
                            for x, tau in zip(buckets, taus)]

        # select n sequences to return
        selections = []
        pts_buckets = clustering.predict(pts_em) # buckets of all unlabeled sequences
        scores = self.embed.predict(pts) # predicted labels for greedy step
        
        for i in range(n):

            # 1. Thompson sample a bucket by sampling from each conjugate dist and taking max
            # 2. get the unlabeled sequences in it and their predictions
            dists = sigma_dists if i / n < self.rho else mu_dists
            samples = np.array([dist() if bucket_idx in pts_buckets else -np.inf
                                        for bucket_idx, dist in enumerate(dists)])
            sampled_idx = np.argmax(samples) == pts_buckets
            sampled_pts = np.array(pts)[sampled_idx]
            sampled_preds = np.array(scores)[sampled_idx]

            # e-greedily take best predicted sequence in bucket
            if np.random.rand() < self.eps:
                selections.append(np.random.choice(sampled_pts))
            else:
                selections.append(sampled_pts[np.argmax(sampled_preds)])

            # remove sequence
            del pts_em[pts.index(selections[-1])]
            removal = np.arange(scores.shape[0]) != pts.index(selections[-1])
            pts_buckets = pts_buckets[removal]
            scores = scores[removal]
            del pts[pts.index(selections[-1])]

        return selections

    def __init__(self, encoder, dim, shape, alpha=5e-4,
                    prior=(0.5, 10, 1, 1), eps=0., rho=0., k=100, minibatch=100):
        '''encoder: convert sequences to one-hot arrays.
        alpha: embedding learning rate.
        shape: sequence shape (len, channels)
        dim: embedding dimensionality
        prior: (mu0, n0, alpha, beta) prior over gamma and gaussian bucket score distributions.
        eps: epsilon for greedy maximization step
        rho: portion of steps to use inverse gamma conjugate instead of normal gamma
        k: cluster count or method
        '''
        super().__init__()
        self.X, self.Y = (), ()
        self.embed = Featurizer(encoder, shape, dim=dim, alpha=alpha, minibatch=minibatch)
        self.prior = prior
        self.eps = eps
        self.rho = rho
        self.k = k


import numpy as np
from random import *
import os, sys
import torch
from torch import nn
import torch.functional as F
from models.featurizer import Featurizer
from sklearn.cluster import KMeans, AffinityPropagation
from sklearn.metrics import silhouette_score


class Combinator:
    '''Buckets sequences and samples from action space consisting of all possible 
    distributions of the batch over the buckets with Thompson sampling, approximating 
    conjugate with MCMC.
    '''

    def fit(self, seqs, scores, epochs):
        '''Fits model to observed labeled sequences. Should be
        called with all labeled sequences seen so far at each
        time step.
        '''
        self.X = seqs[:]
        self.Y = scores[:]
        self.embed.fit(self.X, self.Y, epochs)

    def sample(self, pts, m):
        '''Thompson sample sequences.
        pts: sequences to sample from
        m: number of sequences to sample
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

        def conjugate(x):
            '''Returns conjugate distribution over mean mu and standard deviation
            sigma in provided bucket x.
            '''
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
                return np.random.normal(mu_exp, np.sqrt(mu_var)), np.sqrt(1 / tau)

            return sample

        # construct conjugate distributions for each bucket
        dists = [conjugate(x) for x in buckets]

        def start_state():
            '''Return random k-tuple start state for MCMC corresponding to distribution
            of buckets to sample.
            '''
            idx = [0] + list(1 + np.random.choice(m + k - 1, k - 1, replace=False)) + [m + k]
            return tuple([idx[i + 1] - idx[i] - 1 for i in range(k)])

        def evaluate(state):
            '''Sample value of state using rho value.'''
            values = sorted([np.random.normal(*dists[i]()) for i, v in enumerate(state) 
                                for _ in range(v)], key=lambda x: -x)
            return sum(values[:int(self.rho * m)])

        def transition(state):
            '''Randomly perturb state for MCMC.'''
            num = 1 + np.random.poisson(1)
            new_state = np.array(state)
            for _ in range(num):
                i, j = np.random.choice(k, 2, replace=False)
                if new_state[i] > 0 and new_state[j] < k:
                    new_state[[i, j]] += [-1, 1]
            return tuple(new_state)

        def mcmc(iters)
            '''Approximate action with highest value sampled from conjugate 
            using the provided number of MCMC iterations.
            '''
            state = start_state()
            score = evaluate(state)
            seen = set([state])
            for _ in range(iters):
                next_state = transition(state)
                if next_state in seen:
                    continue
                next_score = evaluate(next_state)
                if next_score > score:
                    state, score = next_state, next_score
                    seen.add(state)

        # get distribution of sequences to sample across buckets with MCMC 
        # approximation of Thompson sampling
        final_state = mcmc(self.iters)
        action = sum([[i] * v for i, v in enumerate(final_state)])
        selections = []
        pts_buckets = clustering.predict(pts_em) # buckets of all unlabeled sequences
        scores = self.embed.predict(pts) # predicted labels for greedy step
        
        # select sequences from buckets
        for bucket_idx in action:

            # 1. Thompson sample a bucket by sampling from each conjugate dist and taking max
            # 2. get the unlabeled sequences in it and their predictions
            sampled_idx = bucket_idx == pts_buckets
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

    def __init__(self, encoder, dim, shape, alpha=5e-4, prior=(0.5, 10, 1, 1), eps=0., 
                        rho=1.0, k=100, iters=1000, minibatch=100):
        '''encoder: convert sequences to one-hot arrays.
        alpha: embedding learning rate
        shape: sequence shape (len, channels)
        dim: embedding dimensionality
        prior: (mu0, n0, alpha, beta) prior over gamma and gaussian bucket score distributions
        eps: epsilon for greedy maximization step
        rho: top portion of sequences to evaluate for MCMC step (should correspond to metric)
        k: cluster count or method
        iters: iterations for MCMC optimization
        '''
        super().__init__()
        self.X, self.Y = (), ()
        self.embed = Featurizer(encoder, shape, dim=dim, alpha=alpha, minibatch=minibatch)
        self.prior = prior
        self.eps = eps
        self.rho = rho
        self.k = k
        self.iters = iters


import numpy as np
from random import sample, choice, random
from scipy.spatial.distance import cdist, pdist, squareform

def cost(pts):
    dist = squareform(pdist(pts))
    dist[np.arange(len(pts)), np.arange(len(pts))] = np.inf
    return np.exp(-dist).sum()

def mcmc(k, em, iters, T=0, lam=1.):
    '''Given points in em, selects k maximally separated points
    approximated by MCMC iteration and returns their indices.
    iters: number of iterations.
    T: temperature parameter
    lam: lambda of poisson distribution used for perturbations
    '''
    pts = range(len(em))
    curr = np.array([*sample(pts, k)])
    best = curr
    c_best = cost(em[best])
    rest = [x for x in pts if x not in curr]
    for i in range(iters):
        idx = sample(range(curr.shape[0]),
                         min(np.random.poisson(lam) + 1, k))
        test = np.array(curr)
        probs = cdist(em[test[idx]], em[rest]) ** (-2)
        probs /= probs.sum(axis=1)[:, None]
        for i, _ in enumerate(idx):
            j = np.random.choice(len(rest), p=probs[i])
            probs[:, j] = 0
            probs /= probs.sum(axis=1)[:, None]
            test[idx[i]] = rest[j]
        c_test = cost(em[test])
        c_curr = cost(em[curr])
        if c_test < c_curr or T != 0 and \
                random() < np.exp((c_test - c_curr) / T):
            curr = test
            rest = [x for x in pts if x not in curr]
            if c_test < c_best:
                best = test
                c_best = c_test
    return best

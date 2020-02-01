import numpy as np
from random import *

def make_motif(sz, comp):
    '''Get Cauchy sampled PWM of provided size and complexity from (0, 1].'''
    arr = np.abs(np.random.normal(size=4 * sz).reshape([sz, 4])) ** (1 / comp)
    return arr / arr.sum(axis=1)[:, np.newaxis]

def seq(m):
    '''Sample a sequence s from PWM m.'''
    return ''.join(np.random.choice(list('ATCG'), p=p) for p in m)

def uniform(n):
    '''Return uniformly sampled sequence of length m.'''
    return choice('+-') + seq(np.array([[0.25] * 4] * (n - 1)))

def pad(D, n=30000):
    '''Add zero-label random sequences to D until |D| = n.'''
    sz = len(D[0][0])
    while len(D) < n:
        D.append((uniform(sz), 0.))

@np.vectorize
def invert(s):
    return 'TAGC'['ATCG'.index(s)]

def gen_seq(length, motifs, lam, scale):
    '''Make sequence of provided length given motifs, mean of poisson 
    distribution for number of motifs in any sequence, and (mu, sigma)
    score distribution pairs for each motif.
    Return the sequence, and the score determined by the quadratic function
    on motif counts specified by the scale matrix.
    '''
    s = np.array([choice('ATCG') for _ in range(length)])
    r = np.zeros(shape=[length + 1 - len(seq(motifs[0]))])
    num = min(1 + np.random.poisson(lam), int(3 * lam + 1))
    features = np.zeros([len(motifs)])
    for i in range(num):
        idx, motif = choice([*enumerate(motifs)])
        m = np.array(list(seq(motif)))
        j = randrange(length + 1 - len(m))
        s[j : j + len(m)] = m
        features[idx] += 1
    sign = choice('+-')
    value = np.random.normal(scale[:, 0], scale[:, 1])
    return sign + ''.join(s if sign == '+' else invert(s)), 1 / (1 + np.exp(-np.dot(value, features) / num))

def make_data(batch, lam=1., N=100, comp=0.5, var=0.5):
    '''Return batch datapoints, each with lam + 1 motifs on average (at least 1).
    Uses a pool of N motifs across the data. Comp
    scales directly with the randomness of the PWMs, var
    is the max variance of scores corresponding to sequences
    with a singe given motif.
    '''
    motifs = [make_motif(10, comp) for _ in range(N)]
    scale = np.array([[random() - 1 / 2, random() * var] for _ in motifs])
    data = [gen_seq(50, motifs, lam, scale) for k in range(batch)]
    shuffle(data)
    return data

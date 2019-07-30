import numpy as np
from random import *

def make_motif(sz):
    '''Get Cauchy sampled PWM of provided size.'''
    arr = np.abs(np.random.standard_cauchy(4 * sz).reshape([sz, 4]))
    return arr / arr.sum(axis=1)[:, np.newaxis]

def seq(m):
    '''Sample a sequense s from PWM m.'''
    return ''.join(np.random.choice(list('ATCG'), p=p) for p in m)

@np.vectorize
def invert(s):
    return 'TAGC'['ATCG'.index(s)]

def gen_seq(length, motifs, rates, lam):
    '''Make sequence of provided length given motifs, motif scores, 
    and mean of poisson distribution for number of motifs in any sequence.
    Return the sequence, and a same-sized array with the scores
    of motifs where they occur in the sequence.
    '''
    s = np.array([choice('ATCG') for _ in range(length)])
    r = np.zeros(shape=[length + 1 - len(seq(motifs[0]))])
    for i in range(np.random.poisson(lam)):
        motif, rate = choice([*zip(motifs, rates)])
        m = np.array(list(seq(motif)))
        j = randrange(length + 1 - len(m))
        s[j : j + len(m)] = m
        r[j] = rate
    sign = choice('+-')
    return sign + ''.join(s if sign == '+' else invert(s)), r

def make_data(batch, lam=3., N=100):
    '''Return batch datapoints, each with lam motifs on average.
    Uses a pool of N motifs across the data.
    '''
    motifs = [(make_motif(10), random()) for _ in range(N)]
    data = []
    for k in range(batch):
        s = gen_seq(50, *zip(*motifs), lam)
        data.append((s[0], 1 / (1 + np.exp(s[1].sum()))))
    return data

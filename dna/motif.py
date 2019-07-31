import numpy as np
from random import *

def make_motif(sz, comp):
    '''Get Cauchy sampled PWM of provided size and complexity from (0, 1].'''
    arr = np.abs(np.random.standard_cauchy(4 * sz).reshape([sz, 4])) ** (1 / comp)
    return arr / arr.sum(axis=1)[:, np.newaxis]

def seq(m):
    '''Sample a sequense s from PWM m.'''
    return ''.join(np.random.choice(list('ATCG'), p=p) for p in m)

@np.vectorize
def invert(s):
    return 'TAGC'['ATCG'.index(s)]

def gen_seq(length, motifs, rates, lam, scale):
    '''Make sequence of provided length given motifs, motif scores, 
    and mean of poisson distribution for number of motifs in any sequence.
    Return the sequence, and a same-sized array with the scores
    of motifs where they occur in the sequence, multiplied by
    the component of scale corresponding to the number of motifs.
    '''
    s = np.array([choice('ATCG') for _ in range(length)])
    r = np.zeros(shape=[length + 1 - len(seq(motifs[0]))])
    num = np.random.poisson(lam)
    if num > 3 * lam: num = int(3 * lam)
    for i in range(num):
        motif, rate = choice([*zip(motifs, rates)])
        m = np.array(list(seq(motif)))
        j = randrange(length + 1 - len(m))
        s[j : j + len(m)] = m
        r[j] = rate * scale[num]
    sign = choice('+-')
    return sign + ''.join(s if sign == '+' else invert(s)), r

def make_data(batch, lam=3., N=100, gamma=5, comp=0.5):
    '''Return batch datapoints, each with lam motifs on average.
    Uses a pool of N motifs across the data. Higher gamma
    corresponds to fewer high-value sequences and comp
    scales directly with the randomness of the PWMs.
    '''
    motifs = [(make_motif(10, comp), random()) for _ in range(N)]
    data = []
    scale = [random() * 2 - 1 for i in range(int(3 * lam + 1))]
    for k in range(batch):
        s = gen_seq(50, *zip(*motifs), lam, scale)
        data.append((s[0], s[1].sum()))
    keys = [x[0] for x in sorted(data, key=lambda d: -d[1])]
    rate, r = 1 / batch, 1.
    results = []
    for k in keys:
        results.append((k, r))
        r *= (1 - rate) ** gamma
    shuffle(results)
    return results

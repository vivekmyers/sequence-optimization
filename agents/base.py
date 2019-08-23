import numpy as np
from random import *
import dna.featurize

class BaseAgent:
    '''Template for agent classes.'''

    def __init__(self, prior, shape, batch, encode):
        self.seen = {}
        self.prior = prior
        self.batch = batch
        self.encode = self._memoize(encode)
        self.shape = shape
    
    def act(self, data):
        '''Return batch of sequences to try.'''
        return sample(data, self.batch)
    
    def observe(self, obs):
        '''Add sequences with known scores to self.seen.'''
        self.seen = {**self.seen, **obs}
    
    def predict(self, seqs):
        '''Predict sequence scores.'''
        return [choice(list(self.seen.values())) for _ in seqs]

    def _memoize(self, f):
        '''Memoize pure function of one parameter.'''
        cache = {}
        def g(x):
            if x in cache:
                return cache[x]
            else:
                cache[x] = f(x)
                return cache[x]
        return g



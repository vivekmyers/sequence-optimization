import numpy as np
from random import *
import dna.featurize

class BaseAgent:
    '''Template for agent classes.'''

    def __init__(self, prior, length, batch):
        self.seen = {}
        self.prior = prior
        self.batch = batch
        self.encode_cache = {}
        self.shape = (length - 1, 4 + dna.featurize.num_features)
        
    def encode(self, seq):
        if seq in self.encode_cache:
            return self.encode_cache[seq]
        assert seq[0] in '+-'
        arr = dna.featurize.encode(seq)
        self.encode_cache[seq] = arr
        return arr
    
    def act(self, data):
        '''Return batch of sequences to try.'''
        return sample(data, self.batch)
    
    def observe(self, obs):
        '''Add sequences with known scores to self.seen.'''
        self.seen = {**self.seen, **obs}
    
    def predict(self, seqs):
        '''Predict sequence scores.'''
        return [choice(list(self.seen.values())) for _ in seqs]


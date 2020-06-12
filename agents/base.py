import numpy as np
from random import *
import environment.featurize


class BaseAgent:
    '''Template for agent classes.'''

    def __init__(self, prior, shape, batch, encode):
        self.seen = {}
        self.prior = prior
        self.batch = batch
        self.encode = encode
        self.shape = shape
        self.seen = {**prior}
    
    def act(self, data):
        '''Return batch of sequences to try.'''
        return sample(data, self.batch)
    
    def observe(self, obs):
        '''Add sequences with known scores to self.seen.'''
        self.seen = {**self.seen, **obs}
    
    def predict(self, seqs):
        '''Predict sequence scores.'''
        return [choice(list(self.seen.values())) for _ in seqs]


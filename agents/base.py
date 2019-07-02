import numpy as np
from random import *

class BaseAgent:
    '''
    Template for agent classes.
    '''

    def __init__(self, prior, length, batch):
        self.seen = prior
        self.batch = batch
        self.encode_cache = {}
        self.embed_cache = {}
        self.len = length
        self.proj = np.random.normal(size=[6, length * 4])
        
    def encode(self, seq):
        if seq in self.encode_cache:
            return self.encode_cache[seq]
        assert seq[0] in '+-'
        arr = np.zeros([len(seq), 4])
        arr[0, :] = 1 if seq[0] == '-' else 0
        arr[(np.arange(1, len(seq)), ['ATCG'.index(i) for i in seq[1:]])] = 1
        self.encode_cache[seq] = arr
        return arr
    
    def embed(self, seq):
        if seq in self.embed_cache:
            return self.embed_cache[seq]
        arr = self.proj @ self.encode(seq).flatten()[:, np.newaxis]
        self.embed_cache[seq] = arr
        return arr
    
    def act(self, data):
        return sample(data, self.batch)
    
    def observe(self, obs):
        self.seen = {**self.seen, **obs}
    
    def predict(self, seqs):
        return [choice(list(self.seen.values())) for _ in seqs]

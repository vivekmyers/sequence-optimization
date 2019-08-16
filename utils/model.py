import numpy as np

def batch(f):
    def method(self, seqs):
        results = []
        while len(seqs):
            results += list(f(self, seqs[:self.minibatch]))
            seqs = seqs[self.minibatch:]
        return np.array(results)
    return method


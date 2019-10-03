import numpy as np

def batch(f):
    '''Decorator on method to evaluate over first argument in minibatches
    of size self.minibatch.
    '''
    def method(self, seqs, *args):
        results = []
        while len(seqs):
            results += list(f(self, seqs[:self.minibatch], *args))
            seqs = seqs[self.minibatch:]
        return np.array(results)
    return method


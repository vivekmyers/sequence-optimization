import numpy as np

def batch(f):
    def method(self, seqs):
        results = []
        seqs = list(seqs)
        while len(seqs):
            results += list(f(self, seqs[-self.minibatch:]))
            del seqs[-self.minibatch:]
        return np.array(results)
    return method


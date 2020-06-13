import time
import numpy as np

def Regret(top):
    '''Computes cumulative difference between sum of the labels of the
    top (top * batch) unobserved sequences and the top (top * batch)
    selected sequences.
    '''
    assert 0 < top <= 1

    class Metric:
        def __init__(self, prior):
            self.history = 0

        def __call__(self, seen, unseen, selected):
            to_check = int(top * len(selected))
            r = sum(sorted(unseen[x] for x in selected)[-to_check:])
            r_star = sum(sorted(unseen.values())[-to_check:])
            self.history += r_star - r
            return self.history

    return Metric


def Score(top):
    '''Computes mean of the top sequence labels seen or selected
    so far at each timestep.
    top: fraction of sequences to evaluate
    '''
    assert 0 < top <= 1

    class Metric:
        def __init__(self, prior):
            pass

        def __call__(self, seen, unseen, selected):
            to_check = int(top * (len(seen) + len(selected)))
            to_score = list(seen.values()) + [unseen[x] for x in selected]
            return np.array(sorted(to_score)[-to_check:]).mean()

    return Metric


def Discovery(top):
    '''Computes fraction of top sequences seen or selected at each timestep.
    top: fraction of sequences to evaluate
    '''
    assert 0 < top <= 1

    class Metric:
        def __init__(self, prior):
            self.best = None

        def __call__(self, seen, unseen, selected):
            if self.best is None:
                all_seqs = {**seen, **unseen}
                num_top = int(top * len(all_seqs))
                self.best = set(sorted(all_seqs.keys(), key=lambda x: all_seqs[x])[-num_top:])
            to_check = list(seen.keys()) + list(selected)
            return sum(x in self.best for x in to_check) / len(self.best)

    return Metric
            

def Time():
    '''Measures cumulative wall-clock time at each timestep.'''

    class Metric:
        def __init__(self, prior):
            self.start = time.time()

        def __call__(self, seen, unseen, selected):
            return time.time() - self.start

    return Metric

def Improvement():
    '''Measures selections until improved sequence found from start sequences.'''

    class Metric:
        def __init__(self, prior):
            self.found = None
            self.best = max(x for x in prior.values())
            self.offset = len(prior)

        def __call__(self, seen, unseen, selected):
            if self.found is not None:
                return self.found
            if any(unseen[x] > self.best for x in selected):
                self.found = len(selected) + len(seen) - self.offset
                return self.found

    return Metric

import Bio.SeqUtils.MeltingTemp as Tm
import Bio.SeqUtils
import numpy as np

num_features = 4

def count_gc(s):
    return np.array([sum(i in 'GC' for i in s) / len(s)]  * (len(s) - 1))

def melting_temp(s):
    return np.array([Tm.Tm_staluc(x, rna=False) for x in s[1:]])

def molecular_weight(s):
    return np.array([Bio.SeqUtils.molecular_weight(x) for x in s[1:]])

def encode(seq):
    arr = np.zeros([len(seq) - 1, 4 + num_features])
    arr[:, 4] = 1 if seq[0] == '-' else 0
    arr[:, 5] = count_gc(seq)
    arr[:, 6] = melting_temp(seq)
    arr[:, 7] = molecular_weight(seq)
    arr[(np.arange(0, len(seq) - 1), ['ATCG'.index(i) for i in seq[1:]])] = 1
    return arr


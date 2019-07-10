import Bio.SeqUtils.MeltingTemp as Tm
import numpy as np

num_features = 2

def count_gc(s):
    return np.array([sum(i in 'GC' for i in s) / len(s)]  * 4)

def melting_temp(s):
    return np.array([Tm.Tm_staluc(x, rna=False) for x in [s[1:7], s[7:12], s[12:18], s[18:]]])

def encode(seq):
    arr = np.zeros([len(seq) + num_features, 4])
    arr[0, :] = 1 if seq[0] == '-' else 0
    arr[1, :] = count_gc(seq)
    arr[2, :] = melting_temp(seq)
    arr[(np.arange(3, len(seq) + num_features), ['ATCG'.index(i) for i in seq[1:]])] = 1
    return arr


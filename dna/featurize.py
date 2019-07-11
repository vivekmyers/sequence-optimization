import Bio.SeqUtils.MeltingTemp as Tm
import Bio.SeqUtils
import numpy as np
import itertools

num_features = 68

def count_gc(s):
    return np.array([sum(i in 'GC' for i in s) / len(s)]  * (len(s) - 1))

def melting_temp(s):
    return np.array([Tm.Tm_staluc(x, rna=False) for x in s[1:]]) / 1e3

def molecular_weight(s):
    return np.array([Bio.SeqUtils.molecular_weight(x) for x in s[1:]]) / 1e3

def kmers(s, k=3):
    '''Returns array of n-hot locations of each kmer in the sequence.'''
    s = s[1:] # remove strand +/-
    features = np.zeros(shape=(4 ** k, len(s)))
    for i, kmer in enumerate(itertools.product('ATCG', repeat=k)):
        for j in range(len(s)):
            if s[j : j + k] == ''.join(kmer):
                features[i][j] = 1
    return features
        
def encode(seq):
    arr = np.zeros([len(seq) - 1, 4 + num_features])
    arr[:, 4] = 1 if seq[0] == '-' else 0
    arr[:, 5] = count_gc(seq)
    arr[:, 6] = melting_temp(seq)
    arr[:, 7] = molecular_weight(seq)
    arr[:, 8 :] = kmers(seq).T
    arr[(np.arange(0, len(seq) - 1), ['ATCG'.index(i) for i in seq[1:]])] = 1
    return arr


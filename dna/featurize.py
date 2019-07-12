import Bio.SeqUtils.MeltingTemp as Tm
import Bio.SeqUtils
import numpy as np
import itertools

# number of additional features in encoded sequences besides 4 bases
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
    val = lambda b: 'ATCG'.index(b)
    for j in range(len(s)):
        kmer = s[j : j + k]
        if len(kmer) == k:
            total = 0
            while len(kmer):
                total = 4 * total + val(kmer[0])
                kmer = kmer[1:]
            features[total][j] = 1.
    return features
        
def encode(seq):
    '''Convert DNA sequence [+-][ATCG]{N} into one-hot array
    with shape [N, 4 + num_features], and with GC frequency,
    melting temperature, molecular weight, and 3mer features 
    appended.
    '''
    arr = np.zeros([len(seq) - 1, 4 + num_features])
    arr[:, 4] = 1 if seq[0] == '-' else 0
    arr[:, 5] = count_gc(seq)
    arr[:, 6] = melting_temp(seq)
    arr[:, 7] = molecular_weight(seq)
    arr[:, 8 :] = kmers(seq).T
    arr[(np.arange(0, len(seq) - 1), ['ATCG'.index(i) for i in seq[1:]])] = 1
    return arr


import Bio.SeqUtils.MeltingTemp as Tm
import Bio.SeqUtils
import numpy as np
import itertools


# number of additional features in encoded DNA sequences 
num_dna_features = 72


# amino acids
aa = ['Aba', 'Ace', 'Acr', 'Ala', 'Aly', 'Arg', 'Asn', 'Asp', 'Cas', 
        'Ccs', 'Cme', 'Csd', 'Cso', 'Csx', 'Cys', 'Dal', 'Dbb', 'Dbu', 
        'Dha', 'Gln', 'Glu', 'Gly', 'Glz', 'His', 'Hse', 'Ile', 'Leu', 
        'Llp', 'Lys', 'Men', 'Met', 'Mly', 'Mse', 'Nh2', 'Nle', 'Ocs', 
        'Pca', 'Phe', 'Pro', 'Ptr', 'Sep', 'Ser', 'Thr', 'Tih', 'Tpo', 
        'Trp', 'Tyr', 'Unk', 'Val', 'Ycm', 'Sec', 'Pyl', 'Ter']
aa_dict = {x: aa.index(x) for x in aa}
num_aa = len(aa)


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
       

def encode_dna(seq):
    '''Convert DNA sequence [+-][ATCG]{N} into one-hot array
    with shape [N, 4 + num_features], and with GC frequency,
    melting temperature, molecular weight, and 3mer features 
    appended.
    '''
    arr = np.zeros([len(seq) - 1, num_dna_features])
    arr[:, 4] = 1 if seq[0] == '-' else 0
    arr[:, 5] = count_gc(seq)
    arr[:, 6] = melting_temp(seq)
    arr[:, 7] = molecular_weight(seq)
    arr[:, 8 :] = kmers(seq).T
    arr[(np.arange(0, len(seq) - 1), ['ATCG'.index(i) for i in seq[1:]])] = 1
    return arr


class ProteinEncoder:

    def __init__(self, base_seq):
        self.base_seq = []
        while len(base_seq):
            self.base_seq.append(base_seq[:3])
            base_seq = base_seq[3:]
        self.base_seq = np.array(self.base_seq)

    def __call__(self, delta):
        '''Return one-hot encoding of base_seq, with the modifications
        in the hgvs_pro delta.
        '''
        result = self.base_seq.copy()
        if delta[:2] == 'p.':
            delta = delta[2:]
            delta = delta.replace('[', '').replace(']', '')
            subs = delta.split(';')
            for s in subs:
                if len(s) > 6:
                    old = s[:3]
                    new = s[-3:]
                    idx = s[3:-3]
                    assert result[int(idx) - 1] == old
                    result[int(idx) - 1] = new
        arr = np.zeros([*self.base_seq.shape, num_aa])
        arr[np.arange(*self.base_seq.shape), [aa_dict[i] for i in result]] = 1.
        return arr
        




import Bio.SeqUtils.MeltingTemp as Tm
import Bio.SeqUtils
import numpy as np
import itertools

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
    with shape [N, 72], and with GC frequency,
    melting temperature, molecular weight, and 3mer features 
    appended.
    '''
    arr = np.zeros([len(seq) - 1, 72])
    arr[:, 4] = 1 if seq[0] == '-' else 0
    arr[:, 5] = count_gc(seq)
    arr[:, 6] = melting_temp(seq)
    arr[:, 7] = molecular_weight(seq)
    arr[:, 8 :] = kmers(seq).T
    arr[(np.arange(0, len(seq) - 1), ['ATCG'.index(i) for i in seq[1:]])] = 1
    return arr


class SeqEncoder:

    features = 5

    def __init__(self, size):
        '''Construct generic ATCG sequence encoder with sequence length.'''
        self.cache = {}
        self.shape = (size, self.features)

    def __call__(self, seq):
        '''Convert DNA sequence [+-][ATCG]{N} into one-hot array
        with shape [N, 5] and +/- strand direction in the last channel.
        '''
        if seq in self.cache:
            idx = self.cache[seq]
        else:
            idx = np.array(['ATCG'.index(i) for i in seq[1:]])
            self.cache[seq] = idx
        arr = np.zeros([len(seq) - 1, self.features])
        assert arr.shape == self.shape and seq[0] in '+-', 'bad sequence'
        arr[:, 4] = 1 if seq[0] == '-' else 0
        arr[(np.arange(0, len(seq) - 1), idx)] = 1
        return arr


class ProteinEncoder:

    aa = ['Aba', 'Ace', 'Acr', 'Ala', 'Aly', 'Arg', 'Asn', 'Asp', 'Cas', 
            'Ccs', 'Cme', 'Csd', 'Cso', 'Csx', 'Cys', 'Dal', 'Dbb', 'Dbu', 
            'Dha', 'Gln', 'Glu', 'Gly', 'Glz', 'His', 'Hse', 'Ile', 'Leu', 
            'Llp', 'Lys', 'Men', 'Met', 'Mly', 'Mse', 'Nh2', 'Nle', 'Ocs', 
            'Pca', 'Phe', 'Pro', 'Ptr', 'Sep', 'Ser', 'Thr', 'Tih', 'Tpo', 
            'Trp', 'Tyr', 'Unk', 'Val', 'Ycm', 'Sec', 'Pyl', 'Ter']

    def __init__(self, base_seq):
        '''Construct protein encoding function with base sequence to be modified.'''
        self.base_seq = []
        while len(base_seq):
            self.base_seq.append(base_seq[:3])
            base_seq = base_seq[3:]
        self.base_seq = np.array(self.base_seq)
        self.shape = (len(self.base_seq), len(self.aa))
        self.cache = {}
        self.aa_dict = {x: self.aa.index(x) for x in self.aa}

    def __call__(self, delta):
        '''Return one-hot encoding of base_seq, with the modifications
        in the hgvs_pro delta.
        '''
        if delta in self.cache:
            result = self.cache[delta]
        else:
            result = self._translate(delta)
            self.cache[delta] = result 
        arr = np.zeros([*self.base_seq.shape, len(self.aa)])
        arr[np.arange(*self.base_seq.shape), [self.aa_dict[i] for i in result]] = 1.
        return arr

    def _translate(self, delta):
        result = self.base_seq.copy()
        if delta[:2] == 'p.':
            delta = delta[2:]
            delta = delta.replace('[', '').replace(']', '')
            subs = delta.split(';')
            for s in set(subs):
                if len(s) > 6 and s[-1] != '=':
                    old = s[:3]
                    new = s[-3:]
                    idx = int(s[3:-3])
                    assert result[idx - 1] == old, idx
                    result[idx - 1] = new
        return np.array(result)

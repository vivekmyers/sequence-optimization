import numpy as np
import pandas as pd
from tqdm import tqdm
from random import *
import math
import pickle
import os
from functools import partial
import gym_batgirl.utils.motif as motif
import gym_batgirl.utils.featurize as featurize
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import pkg_resources

DATA_PATH = pkg_resources.resource_filename('gym_batgirl', 'data/')

class Batgirl(gym.Env):
    '''Environment for all sequence data.'''

    metadata = {'render.modes': []}

    def __init__(self):
        super().__init__()
        self.initialized = False

    def step(self, action):
        '''Given action consisting of self.batch unlabeled sequences, return observed 
        labels, regret, done state, and remaining actions.
        '''
        assert self.initialized, 'environment must be reset()'
        assert all(a in self.data for a in action) and len(set(action)) == len(action) == self.batch, 'bad action'
        assert len(self.data) >= self.batch, 'done'
        obs = {a: self.data[a] for a in action}
        r_star = np.array(sorted([v for k, v in self.data.items()]))[math.floor(-self.metric * self.batch):].mean()
        r = np.array(sorted([v for k, v in obs.items()]))[math.floor(-self.metric * self.batch):].mean()
        self.data = {k: v for k, v in self.data.items() if k not in set(action)}
        regret = r_star - r
        done = len(self.data) < self.batch
        info = [k for k, v in self.data.items()]
        return obs, regret, done, info

    def reset(self, src, metric=0.2, batch=100):
        '''Reset environment with given sequence data source.
        metric: portion of sequences for regret computation
        batch: action size
        src: one of "cluster-{N}-{comp}-{var}", "motif-{N}-{comp}-{var}", 
                        "protein-{name}", "primer-{20|30}", "primer", "mpra", "guide", "flank"
        '''
        assert 0 < metric <= 1 and batch > 0 
        arg = (batch, 0.0)
        self.metric = metric
        self.batch = batch

        if src.startswith('cluster'):
            _, a, b, c = src.split('-')
            env = ClusterEnv(int(a), float(b), float(c))(*arg)
            env._make_data()
        elif src.startswith('motif'):
            _, a, b, c = src.split('-')
            env = MotifEnv(int(a), float(b), float(c))(*arg)
            env._make_data()
        elif src.startswith('protein'):
            _, p = src.split('-')
            env = ProteinEnv(p)(*arg)
        elif src.startswith('primer'):
            _, p = src.split('-') if '-' in src else (None, None)
            env = PrimerEnv(int(p))(*arg) if p else PrimerEnv()(*arg)
        elif src == 'mpra':
            env = NormalizedMPRAEnv(*arg)
        elif src == 'guide':
            env = GuideEnv(*arg)
        elif src == 'flank':
            env = FlankEnv(*arg)
        else:
            raise ValueError('bad data src')
            
        self.data = env.env
        self.encoder = env.encode
        self.initialized = True
        return [k for k, v in self.data.items()]

    def encode(self, seq):
        '''Convert sequence to tensor in environment.'''
        assert self.initialized, 'environment must be reset()'
        return self.encoder(seq)


class _Env:
    '''Stores labeled sequences, reserving some for validation, and runs agents on them. Should be extended with custom
    constructor to set up data.
    '''

    metric = 5 # fraction of sequences to use for regret and reward

    def run(self, Agent, cutoff, name, pos):
        '''Run agent, getting batch-sized list of actions (sequences) to try,
        and calling observe with the labeled sequences until all sequences
        have been tried (or the batch number specified by the cutoff parameter
        has been reached). Returns the validation performance after each batch
        (measured using Pearson correlation with the agent's predict method 
        on validation data), as well as the total performance of the best 
        20% of guides the agent has seen after each batch, and the regret after
        each batch (difference between best possible selection and
        actual top 20% of selection). The name and pos parameters are used for a progress bar.
        '''
        data = self.env.copy()
        if cutoff is None:
            cutoff = 1 + len(data) // self.batch
        pbar = tqdm(total=min(len(data) // self.batch * self.batch, cutoff * self.batch), 
                        position=pos, desc=name)
        agent = Agent(self.prior.copy(), self.shape, self.batch, self.encode)
        seen = []
        corrs = []
        reward = []
        regret = []
        while len(data) >= self.batch and (cutoff is None or len(reward) < cutoff):
            sampled = agent.act(list(data.keys()))
            assert len(set(sampled)) == self.batch, "bad action"
            agent.observe({seq: data[seq] for seq in sampled})
            r_star = sum(sorted(data.values())[-self.batch // self.metric:])
            r = sum(sorted([data[seq] for seq in sampled])[-self.batch // self.metric:])
            regret.append(([0.] + regret)[-1] + r_star - r)
            for seq in sampled:
                seen.append(data[seq])
                del data[seq]
            if not self.nocorr:
                predicted = np.array(agent.predict(self.val[0].copy()))
                corrs.append(np.nan_to_num(np.corrcoef(predicted, self.val[1])[0, 1]))
            reward.append(np.array(sorted(seen))[-len(seen) // self.metric:].mean())
            pbar.update(self.batch)
        pbar.close()
        return np.array(corrs), np.array(reward), np.array(regret)

    def __init__(self, batch, validation, pretrain=False, nocorr=False):
        '''Initialize environment.
        batch: number of sequences selected per action
        validation: portion of sequences to save for validation
        initial: pretrain on given datafile
        nocorr: do not compute correlations when running

        Subclasses must override and set self.env, self.val, self.prior, self.shape, and self.encode before each run.
        self.env: {X: Y ...} data dictionary for running
        self.val: (X, Y) validation data
        self.prior: initial data to pass to agent
        self.shape: encoded sequence shape
        self.encode: convert sequence to tensor
        '''
        assert 0 <= validation < 1
        self.batch = batch
        self.pretrain = pretrain
        self.validation = validation
        self.nocorr = nocorr
        self.cache = {}
        self.prior = {}


class GuideEnv(_Env):
    '''CRISPR guide environment with on-target labels.'''

    def __init__(self, batch, validation, pretrain=False, nocorr=False):
        super().__init__(batch, validation, pretrain, nocorr)
        files=[f'{DATA_PATH}/DeepCRISPR/{f}' for f in os.listdir(f'{DATA_PATH}/DeepCRISPR') if f.endswith('.csv')]
        initial = f'{DATA_PATH}/Azimuth/azimuth_preds.csv.gz' if pretrain else None
        dfs = list(map(pd.read_csv, files))
        data = [(strand + seq, score) for df in dfs
            for _, strand, seq, score in 
            df[['Strand', 'sgRNA', 'Normalized efficacy']].itertuples()]
        shuffle(data)
        self.encode = featurize.SeqEncoder(len(data[0][0]) - 1)
        self.shape = self.encode.shape
        r = int(validation * len(data))
        self.env = dict(data[r:])
        self.val = tuple(np.array(x) for x in zip(*data[:r]))
        # start agent with some portion of initial sequences
        if initial:
            df = pd.read_csv(initial, delimiter=r'\t', engine='python', compression='gzip')
            azimuth_guides = ['+' + s[6:-3].upper() for s, pam in zip(df.guide_seq, df.pam_seq) if pam == 'GG']
            self.prior = dict([*zip(azimuth_guides, df.azimuth_pred)])
        assert batch < len(self.env)


class FlankEnv(_Env):
    '''Simpler environment using flanking sequences.'''

    def __init__(self, batch, validation, pretrain=False, nocorr=False):
        super().__init__(batch, validation, pretrain, nocorr)
        df = pickle.load(open(f'{DATA_PATH}/flanking_sequences/cbf1_reward_df.pkl', 'rb'))
        data = [*zip([f'+{x}' for x in df.index], df.values)]
        shuffle(data)
        dlen = 30000
        self.prior = dict(data[dlen:]) if pretrain else {}
        data = data[:dlen]
        r = int(dlen * validation)
        self.env = dict(data[r:])
        self.val = tuple(np.array(x) for x in zip(*data[:r]))
        self.encode = featurize.SeqEncoder(len(data[0][0]) - 1)
        self.shape = self.encode.shape


class _GenericEnv(_Env):

    def __init__(self, data, prior, batch, validation, pretrain=False, nocorr=False):
        super().__init__(batch, validation, pretrain, nocorr)
        data = pd.read_csv(data, header=None)
        prior = pd.read_csv(prior, header=None) if pretrain and prior is not None else None
        self.prior = dict(prior.values if prior is not None else [])
        data = data.values
        r = int(len(data) * validation)
        self.env = dict(data[r:])
        self.val = tuple(np.array(x) for x in zip(*data[:r]))
        self.encode = featurize.SeqEncoder(len(data[0][0]) - 1)
        self.shape = self.encode.shape


def GenericEnv(data, prior=None):
    '''Parameterized environment built with arbitrary csv [dna sequence, score]
    data and pretraining data files.
    '''
    return partial(_GenericEnv, data, prior)


class _MotifEnv(GuideEnv):

    def __init__(self, N, lam, comp, var, batch, validation, pretrain=False, nocorr=False):
        super().__init__(batch, validation, pretrain, nocorr)
        self.N = N
        self.lam = lam
        self.comp = comp
        self.var = var

    def _make_data(self, dlen=30000):
        data = motif.make_data(dlen, N=self.N, lam=self.lam, comp=self.comp, var=self.var)
        r = int(len(data) * self.validation)
        self.env = dict(data[r:])
        self.val = tuple(np.array(x) for x in zip(*data[:r]))
        self.encode = featurize.SeqEncoder(len(data[0][0]) - 1)
        self.shape = self.encode.shape

    def run(self, *args, **kwargs):
        self._make_data()
        return super().run(*args, **kwargs)


def MotifEnv(N=100, lam=1., comp=0.5, var=0.5):
    '''Parameterized environment with sequences containing on average
    lam motifs (which determine its scores). N motifs are present across
    all sequences in the environment.
    comp: scales with stochasticity of PWMs used to make motifs.
    var: max motif score variance
    '''
    return partial(_MotifEnv, N, lam, comp, var)


class _ClusterEnv(_Env):

    def __init__(self, N, comp, var, batch, validation, pretrain=False, nocorr=False):
        super().__init__(batch, validation, pretrain, nocorr)
        self.encode = featurize.SeqEncoder(20)
        self.shape = self.encode.shape
        self.N = N
        self.comp = comp
        self.var = var

    def _make_data(self, dlen=30000):
        motifs = [(motif.make_motif(self.shape[0], self.comp), 
            random() - 1 / 2, random() * self.var) for _ in range(self.N)]
        data = [(choice('+-') + motif.seq(m), 1 / (1 + np.exp(-np.random.normal(mu, sigma))))
                    for m, mu, sigma in motifs for _ in range(dlen // self.N)]
        shuffle(data)
        r = int(len(data) * self.validation)
        self.env = dict(data[r:])
        self.val = tuple(np.array(x) for x in zip(*data[:r]))

    def run(self, *args, **kwargs):
        self._make_data()
        return super().run(*args, **kwargs)


def ClusterEnv(N=100, comp=0.5, var=0.5):
    '''Parameterized environment with sequences in N clusters all with the
    same motif PWM.
    comp: scales with stochasticity of PWMs used to make motifs.
    var: max variance of score distribution of any cluster.
    '''
    return partial(_ClusterEnv, N, comp, var)


class _ProteinEnv(_Env):
    
    def __init__(self, source, batch, validation, pretrain=False, nocorr=False):
        super().__init__(batch, validation, pretrain, nocorr)
        base_seq = open(f'{DATA_PATH}/MaveDB/seqs/{source}.txt').read().strip()
        df = pd.read_csv(f'{DATA_PATH}/MaveDB/scores/{source}.csv.gz', delimiter=r',', engine='python', compression='gzip')
        data = [(x, y) for x, y in zip(df.hgvs_pro.values, 1 / (1 + np.exp(-df.score.values))) if not np.isnan(y)]
        shuffle(data)
        r = int(len(data) * validation)
        self.env = dict(data[r:])
        self.val = tuple(np.array(x) for x in zip(*data[:r]))
        self.encode = featurize.ProteinEncoder(base_seq)
        self.shape = self.encode.shape


def ProteinEnv(source):
    '''Parameterized environment with MaveDB binding affinity scores for protein sequences.
    data: use files {DATA_PATH}/MaveDB/scores/{source}.csv.gz and {DATA_PATH}/MaveDB/seqs/{source}.txt
    '''
    return partial(_ProteinEnv, source)


class _PrimerEnv(_Env):

    def __init__(self, mer, batch, validation, pretrain=False, nocorr=False):
        super().__init__(batch, validation, pretrain, nocorr)
        df = pd.read_csv(f'{DATA_PATH}/primers/primers.txt', delimiter='\t')
        xcol = {None: 'probe', 20: 'probe_20mer', 30: 'probe_30mer'}
        assert mer in xcol, 'valid options are {None, 20, 30}'
        data = [('+' + x, y) for x, y in zip(df[xcol[mer]], df.frac_on_target)]
        shuffle(data)
        r = int(len(data) * validation)
        self.env = dict(data[r:])
        self.val = tuple(np.array(x) for x in zip(*data[:r]))
        self.encode = featurize.SeqEncoder(len(data[0][0]) - 1)
        self.shape = self.encode.shape


def PrimerEnv(mer=None):
    '''Parameterized environment of primer sequences with on target scores.
    mer: One of {None, 20, 30} for the primer sequence length.
    '''
    return partial(_PrimerEnv, mer)


class MPRAEnv(_Env):
    '''MPRA sequences scored by average expression.'''

    def __init__(self, batch, validation, pretrain=False, nocorr=False):
        super().__init__(batch, validation, pretrain, nocorr)
        files = [f'{DATA_PATH}/MPRA/mpra_endo_scramble.txt',
                 f'{DATA_PATH}/MPRA/mpra_endo_tss_lb.txt',
                 f'{DATA_PATH}/MPRA/mpra_peak_tile.txt']
        dfs = [pd.read_csv(f, delimiter='\t') for f in files]
        data = self.normalize([('+' + x, y) for df in dfs for x, y in zip(df.trimmed_seq, df.RNA_exp_ave)])
        shuffle(data)
        r = int(len(data) * validation)
        self.env = dict(data[r:])
        self.val = tuple(np.array(x) for x in zip(*data[:r]))
        self.encode = featurize.SeqEncoder(len(data[0][0]) - 1)
        self.shape = self.encode.shape

    def normalize(self, data):
        maxval = max([y for x, y in data])
        return [(x, y / maxval) for x, y in data]


class NormalizedMPRAEnv(MPRAEnv):
    '''Normalize MPRA scores by making score proportional to rank.'''
    
    def normalize(self, data):
        x_sort = [x for x, y in sorted(data, key=lambda d: d[1])]
        return [(x, i / len(x_sort)) for i, x in enumerate(x_sort)]

import numpy as np
import pandas as pd
from tqdm import tqdm
from random import *
import math
import pickle
import os
from functools import partial
import gym_batgirl.utils.metrics as metrics
from gym_batgirl.utils.env import * 
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
        act_set = set(action)
        assert self.initialized, 'environment must be reset()'
        assert all(a in self.data for a in action) and len(act_set) == len(action) == self.batch, 'bad action'
        assert len(self.data) >= self.batch, 'done'
        regret = self.metric(self.seen, self.unseen, action)
        obs = {a: self.data[a] for a in action}
        self.seen.update(obs)
        self.data = {k: v for k, v in self.data.items() if k not in act_set}
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
        self.metric = metrics.Regret(metric)
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
        self.seen = {}
        self.encoder = env.encode
        self.initialized = True
        return [k for k, v in self.data.items()]

    def encode(self, seq):
        '''Convert sequence to tensor in environment.'''
        assert self.initialized, 'environment must be reset()'
        return self.encoder(seq)



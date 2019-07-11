import numpy as np
import pandas as pd
from tqdm import tqdm
from random import *

class GuideEnv:
    '''Stores labeled sequences, reserving some for validation, and runs agents on them.'''

    def __init__(self, files, batch=1000, validation=0.2, initial=None):
        '''Initialize environment.
        files: csv files to read data from
        batch: number of sequences selected per action
        validation: portion of sequences to save for validation
        initial: pretrain on given datafile
        '''
        assert 0 <= validation < 1
        dfs = list(map(pd.read_csv, files))
        data = [(strand + seq, score) for df in dfs
            for _, strand, seq, score in 
            df[['Strand', 'sgRNA', 'Normalized efficacy']].itertuples()]
        self.len = len(data[0][0])
        shuffle(data)
        r = int(validation * len(data))
        self.env = dict(data[r:])
        self.val = tuple(np.array(x) for x in zip(*data[:r]))
        # start agent with some portion of initial sequences
        if initial:
            df = pd.read_csv(initial, delimiter=r'\t', engine='python', compression='gzip')
            azimuth_guides = ['+' + s[6:-3].upper() for s, pam in zip(df.guide_seq, df.pam_seq) if pam == 'GG']
            self.prior = dict([*zip(azimuth_guides, df.azimuth_pred)])
        else:
            self.prior = {}
        self.batch = batch
        assert batch < len(self.env)
        
    def run(self, Agent, cutoff=None):
        '''Run agent, getting batch-sized list of actions (sequences) to try,
        and calling observe with the labeled sequences until all sequences
        have been tried (or the batch number specified by the cutoff parameter
        has been reached). Returns the validation performance after each batch
        (measured using Pearson correlation with the agent's predict method 
        on validation data), as well as the average performance of the best 
        10 guides the agent has seen after each batch.
        '''
        data = self.env.copy()
        if cutoff is None:
            cutoff = 1 + len(data) // self.batch
        agent = Agent(self.prior.copy(), self.len, self.batch)
        seen = []
        corrs = []
        top10 = []
        pbar = tqdm(total=min(len(data) // self.batch * self.batch, cutoff * self.batch))
        while len(data) > self.batch and (cutoff is None or len(corrs) < cutoff):
            sampled = agent.act(list(data.keys()))
            assert len(set(sampled)) == self.batch, "bad action"
            agent.observe({seq: data[seq] for seq in sampled})
            for seq in sampled:
                seen.append(data[seq])
                del data[seq]
            predicted = np.array(agent.predict(self.val[0].copy()))
            corrs.append(np.corrcoef(predicted, self.val[1])[0, 1])
            top10.append(np.array(sorted(seen))[-10:].mean())
            pbar.update(self.batch)
        pbar.close()
        return np.array(corrs), np.array(top10)
    

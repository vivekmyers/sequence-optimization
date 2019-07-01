import numpy as np
import pandas as pd
from tqdm import tqdm
from random import *

class GuideEnv:
    def __init__(self, files, batch=1000, validation=0.2, initial=0.0):
        dfs = list(map(pd.read_csv, files))
        data = [(strand + seq, score) for df in dfs
            for _, strand, seq, score in 
            df[['Strand', 'sgRNA', 'Normalized efficacy']].itertuples()]
        self.len = len(data[0][0])
        shuffle(data)
        r = int(validation * len(data))
        self.env = dict(data[r:])
        self.val = tuple(np.array(x) for x in zip(*data[:r]))
        self.prior = dict(sample(self.env.items(), int(initial * len(data))))
        self.batch = batch
        
    def run(self, Agent):
        agent = Agent(self.prior.copy(), self.len, self.batch)
        data = self.env.copy()
        results = []
        pbar = tqdm(total=len(data) // self.batch * self.batch)
        while len(data) > self.batch:
            sampled = agent.act(list(data.keys()))
            agent.observe({seq: data[seq] for seq in sampled})
            for seq in sampled:
                del data[seq]
            predicted = np.array(agent.predict(self.val[0].copy()))
            results.append(np.corrcoef(predicted, self.val[1])[0, 1])
            pbar.update(self.batch)
        pbar.close()
        return np.array(results)
    


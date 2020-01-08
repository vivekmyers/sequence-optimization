#!/usr/bin/env python3
import os, sys
sys.path.append('..')
os.chdir('results')
from itertools import *
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')


if __name__ == '__main__':
    '''Produce plot of non-random agents' final regrets for a given run with uncertainty.
    Takes output directory of a run in results/ and places plot there.
    '''

    for i in range(1, len(sys.argv)):
        plt.figure()
        ddir = sys.argv[i]
        os.chdir(ddir)
        try:
            data = np.load('results.npy', allow_pickle=True)
            dmap = [(x['agent'], x['regret'], x['time']) for x in data if not x['agent'].startswith('RandomAgent')]
            grp = [(k, list(g)) for k, g in groupby(dmap, lambda x: x[0])]
            collected = [(k, np.array([y for x, y, z in g]), np.array([z for x, y, z in g])) for k, g in grp]
            dist = [(a, b.mean(axis=0), 2 * b.std(axis=0) / np.sqrt(len(b)), c.mean(axis=0)) for a, b, c in collected]
            
            for i, (name, mu, sigma, t) in enumerate(dist):
                p = plt.plot(t, mu, label=name)
                plt.fill_between(t, mu - sigma, mu + sigma, color=p[0].get_color(), alpha=0.2)

            plt.ylabel('Regret')
            plt.xlabel('Time')
            plt.title(f'{data[0]["env"]},batch={data[0]["batch"]}')
            plt.legend()
            plt.tight_layout()
            plt.savefig(f'time.png', dpi=500)
            os.chdir('..')
        except Exception as e:
            os.chdir('..')
            print(f'failed to process: {sys.argv[i]}')


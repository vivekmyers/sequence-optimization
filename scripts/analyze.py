#!/usr/bin/env python3
import os, sys
sys.path.append('..')
os.chdir('results')
from itertools import *
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')

N = 100
scale = 5

if __name__ == '__main__':
    '''Produce plot of non-random agents' final regrets for a given run with uncertainty.
    Takes output directory of a run in results/ and places plot there.
    '''

    for i in range(1, len(sys.argv)):
        plt.figure()
        ddir = sys.argv[i]
        os.chdir(ddir)
        data = np.load('results.npy', allow_pickle=True)
        dmap = [(x['agent'], x['regret'][-1]) for x in data if not x['agent'].startswith('RandomAgent')]
        collected = [(k, np.array([y for x, y in list(g)])) for k, g in groupby(dmap, lambda x: x[0])]
        dist = [(a, b.mean(), scale * b.std() / np.sqrt(len(b))) for a, b in collected]
        
        for i, (name, mu, sigma) in enumerate(dist):
            p = None
            for k in np.arange(N):
                dense = np.exp(-(k / N * scale) ** 2 / 2)
                X = [4 * i + 1, 4 * i + 1]
                Y = [mu - sigma * k / N, mu + sigma * k / N]
                p = plt.plot(X, Y, label=name, 
                        linewidth=dense * 22, alpha=dense / 5, 
                        color=p[0].get_color() if p else None, solid_capstyle="butt")
        plt.plot([4 * len(dist) - 2], [0])

        plt.xticks(4 * np.arange(len(dist)) - 1, labels=[name for name, x, y in dist], rotation=45)
        plt.ylabel('Final Normalized Regret')
        plt.title(data[0]['env'])
        plt.tight_layout()
        plt.savefig(f'final.png', dpi=500)
        os.chdir('..')


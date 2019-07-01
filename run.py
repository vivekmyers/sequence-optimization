import os, argparse
import env
import seaborn as sns
import matplotlib.pyplot as plt
import importlib
import numpy as np
sns.set_style('darkgrid')

files = [f for f in os.listdir('agents') if f.endswith('.py')]

for f in files :
    mod = importlib.import_module('agents.' + f[:-3])
    globals().update(mod.__dict__)

parser = argparse.ArgumentParser(description='Run flags')
parser.add_argument('--agents', nargs='*', type=str, help='Agent class to use')
parser.add_argument('--batch', type=int, default=1000, help='Batch size')
parser.add_argument('--initial', type=float, default=0.0, help='Starting data portion')
parser.add_argument('--validation', type=float, default=0.2, help='Validation data portion')

args = parser.parse_args()

env = env.GuideEnv(batch=args.batch, initial=args.initial, validation=args.validation,
        files=[f'data/{f}' for f in os.listdir('data') if f.endswith('.csv')])

for agent in args.agents:
    print()
    print(f'Running {agent}...')
    results = env.run(eval(agent))
    print(f'Saving to results/{agent}...')
    plt.figure()
    plt.title(f'{agent}, Batch {args.batch}')
    plt.xlabel('Batch')
    plt.ylabel('Correlation')
    plt.plot(results)
    try: os.mkdir(f'results/{agent}')
    except OSError: pass
    plt.savefig(f'results/{agent}/{agent}-{args.batch}.png')
    np.save(f'results/{agent}/{agent}-{args.batch}.npy', results)
    print()



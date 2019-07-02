import os, argparse
import env
import seaborn as sns
import matplotlib.pyplot as plt
import importlib
import numpy as np
sns.set_style('darkgrid')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


if __name__ == '__main__':

    # Load agent modules
    files = [f for f in os.listdir('agents') if f.endswith('.py')]
    mods = {}

    for f in files :
        mod = importlib.import_module('agents.' + f[:-3])
        mods.update(mod.__dict__)

    # Parse environment parameters
    parser = argparse.ArgumentParser(description='Run flags')
    parser.add_argument('--agents', nargs='*', type=str, help='Agent class to use')
    parser.add_argument('--batch', type=int, default=1000, help='Batch size')
    parser.add_argument('--initial', type=float, default=0.2, help='Starting data portion')
    parser.add_argument('--validation', type=float, default=0.2, help='Validation data portion')

    args = parser.parse_args()

    env = env.GuideEnv(batch=args.batch, initial=args.initial, validation=args.validation,
            files=[f'data/{f}' for f in os.listdir('data') if f.endswith('.csv')])

    # Run agents
    for agent in args.agents:
        print()
        print(f'Running {agent}...')
        results = env.run(eval(agent, mods, {}))
        print(f'Saving to results/{agent}...')
        plt.figure()
        plt.title(f'{agent}, batch={args.batch}')
        plt.xlabel('Batch')
        plt.ylabel('Correlation')
        plt.plot(results)
        try: os.mkdir(f'results/{agent}')
        except OSError: pass
        plt.savefig(f'results/{agent}/{agent}-batch={args.batch}.png')
        np.save(f'results/{agent}/{agent}-batch={args.batch}.npy', results)
        print()



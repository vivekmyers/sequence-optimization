import matplotlib
matplotlib.use('Agg')
import os, argparse
import seaborn as sns
import matplotlib.pyplot as plt
import importlib
import numpy as np
import multiprocessing
import torch
import signal
import dna.env
sns.set_style('darkgrid')
signal.signal(signal.SIGINT, lambda x, y: exit(1))
np.seterr(divide='ignore', invalid='ignore')


def make_plot(title, yaxis, data, loc):
    '''Make a plot of [(Datum, Label)] data and save to given location in results.'''
    plt.figure()
    plt.title(title)
    plt.xlabel('Batch')
    plt.ylabel(yaxis)
    for datum, label in data:  
        if label is None:
            plt.plot(datum)
        else:
            plt.plot(datum, label=label)
    if any([label is not None for _, label in data]):
        plt.legend()
    plt.savefig(f'results/{loc}.png')


def run_agent(arg):
    '''Run agent in provided environment, with given arguments.'''
    env, agent, pos, args = arg
    if not torch.cuda.is_available() and pos == 0: print('CUDA not available')
    name = agent + ' ' * (max(map(len, args.agents)) - len(agent))
    if torch.cuda.is_available():
        with torch.cuda.device(pos % torch.cuda.device_count()):
            corrs, reward, regret = env.run(eval(agent, mods, {}), args.cutoff, name, pos)
    else:
        corrs, reward, regret = env.run(eval(agent, mods, {}), args.cutoff, name, pos)
    data = dict(
        env=args.env,
        agent=agent,
        batch=args.batch,
        pretrain=args.pretrain,
        validation=args.validation,
        correlations=corrs,
        reward=reward,
        regret=regret)
    return dict(agent=agent, corrs=corrs, reward=reward, regret=regret, data=data)


if __name__ == '__main__':

    # Parse environment parameters
    parser = argparse.ArgumentParser(description='run flags')
    parser.add_argument('--agents', nargs='+', type=str, help='agent classes to use', required=True)
    parser.add_argument('--batch', type=int, default=1000, help='batch size')
    parser.add_argument('--cutoff', type=int, default=None, help='max number of batches to run')
    parser.add_argument('--pretrain', action='store_true', help='pretrain on azimuth data')
    parser.add_argument('--validation', type=float, default=0.2, help='validation data portion')
    parser.add_argument('--nocorr', action='store_true', help='do not compute prediction correlations')
    parser.add_argument('--env', type=str, default='GuideEnv', help='environment to run agents')
    parser.add_argument('--reps', type=int, default=1, help='number of trials to average')
    parser.add_argument('--name', type=str, default=None, help='output directory')
    parser.add_argument('--cpus', type=int, default=multiprocessing.cpu_count(), help='number of agents to run concurrurently')

    args = parser.parse_args()

    env = eval(f'{args.env}', dna.env.__dict__, {})(batch=args.batch, validation=args.validation, pretrain=args.pretrain, nocorr=args.nocorr)

    # Load agent modules
    files = [f for f in os.listdir('agents') if f.endswith('.py')]
    mods = {}

    for f in files :
        mod = importlib.import_module('agents.' + f[:-3])
        mods.update(mod.__dict__)

    # Run agents
    thunks = [(env, agent, i * args.reps + j, args)
                            for i, agent in enumerate(args.agents)
                            for j in range(args.reps)]
    pool = multiprocessing.Pool(processes=args.cpus, maxtasksperchild=1)
    collected = pool.map(run_agent, thunks, chunksize=1)

    # Write output
    loc = ",".join(args.agents) if args.name is None else args.name
    try: os.mkdir(f'results/{loc}')
    except OSError: pass
    np.save(f'results/{loc}/results.npy', 
                [x['data'] for x in collected])
    if not args.nocorr:
        corrs = {}
        for agent in [x['agent'] for x in collected]:
            corrs[agent] = np.array([x['corrs'] for x in collected if x['agent'] == agent]).mean(axis=0)
        make_plot(f'batch={args.batch}, env={args.env}, reps={args.reps}', 'Correlation', 
                    [(datum, agent) for agent, datum in corrs.items()],
                    f'{loc}/corr')
    rewards = {}
    for agent in [x['agent'] for x in collected]:
        rewards[agent] = np.array([x['reward'] for x in collected if x['agent'] == agent]).mean(axis=0)
    make_plot(f'batch={args.batch}, env={args.env}, reps={args.reps}', 'Reward', 
                [(datum, agent) for agent, datum in rewards.items()],
                f'{loc}/reward')
    regrets = {}
    for agent in [x['agent'] for x in collected]:
        regrets[agent] = np.array([x['regret'] for x in collected if x['agent'] == agent]).mean(axis=0)
    make_plot(f'batch={args.batch}, env={args.env}, reps={args.reps}', 'Regret', 
                [(datum, agent) for agent, datum in regrets.items()],
                f'{loc}/regret')


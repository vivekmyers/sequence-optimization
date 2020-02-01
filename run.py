import matplotlib
matplotlib.use('Agg')
import os
import argparse
import shutil
import seaborn as sns
import matplotlib.pyplot as plt
import importlib
import traceback
import numpy as np
import multiprocessing
import torch
import signal
import environment.env
import random
import time
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
    env, agent, pos, args, seed, loc = arg
    if not torch.cuda.is_available() and pos == 0: print('CUDA not available')
    name = agent + ' ' * (max(map(len, args.agents)) - len(agent))
    try:
        random.seed(seed[1])
        np.random.seed(seed[1])
        torch.manual_seed(seed[1])
        if torch.cuda.is_available():
            with torch.cuda.device(pos % torch.cuda.device_count()):
                metrics = env.run(eval(agent, mods, {}), args.cutoff, args.metrics, name, pos)
        else:
            metrics = env.run(eval(agent, mods, {}), args.cutoff, args.metrics, name, pos)
    except: 
        traceback.print_exc()
        return None
    data = dict(
        env=args.env,
        agent=agent,
        batch=args.batch,
        validation=args.validation,
        metrics=metrics,
        seed=seed)
    np.save(f'results/{loc}/partial/{agent}-{pos}.npy', data)
    return data


def process_data(attr, collected):
    '''Make plot of metric attr averaged for each agent in the list of collected run results.'''
    results = {}
    for agent in [x['agent'] for x in collected]:
        data = [x['metrics'][attr] for x in collected if x['agent'] == agent]
        n = max(map(len, data))
        def pad(x, n):
            x = list(x)
            while len(x) < n:
                x.append(np.nan)
            return x
        results[agent] = np.array([pad(x, n) for x in data]).mean(axis=0)
    make_plot(f'batch={args.batch}, env={args.env}, reps={args.reps}', attr, 
                [(datum, agent) for agent, datum in results.items()],
                f'{loc}/plots/{attr}')


def get_result(result, timeout):
    '''Get result from async_result if it completes before timeout.'''
    if timeout < 1:
        return None
    result.wait(timeout=int(timeout))
    if result.ready() and result.successful():
        return result.get()


if __name__ == '__main__':

    # Parse environment parameters
    parser = argparse.ArgumentParser(description='run flags')
    parser.add_argument('--agents', nargs='+', type=str, help='agent classes to use', required=True)
    parser.add_argument('--metrics', nargs='+', type=str, help='metrics to evaluate at each timestep', required=True)
    parser.add_argument('--batch', type=int, default=100, help='batch size')
    parser.add_argument('--cutoff', type=int, default=None, help='max number of batches to run')
    parser.add_argument('--validation', type=float, default=0.2, help='validation data portion')
    parser.add_argument('--env', type=str, default='GuideEnv', help='environment to run agents')
    parser.add_argument('--reps', type=int, default=1, help='number of trials to average')
    parser.add_argument('--name', type=str, default=None, help='output directory')
    parser.add_argument('--cpus', type=int, default=multiprocessing.cpu_count(), help='number of agents to run concurrently')
    parser.add_argument('--timeout', type=int, default=36000, help='max time to run agents in seconds')
    parser.add_argument('--seed', type=int, default=None, help='random seed')

    args = parser.parse_args()

    if args.seed is not None:
        seed = args.seed
    else:
        seed = random.randint(0, (1 << 32) - 1)
    random.seed(seed)

    env = eval(f'{args.env}', environment.env.__dict__, {})(batch=args.batch, validation=args.validation)

    # Load agent modules
    files = [f for f in os.listdir('agents') if f.endswith('.py')]
    mods = {}

    for f in files :
        mod = importlib.import_module('agents.' + f[:-3])
        mods.update(mod.__dict__)

    # Make output directory
    loc = ",".join(args.agents) if args.name is None else args.name
    assert len(loc) > 0
    shutil.rmtree(f'results/{loc}', ignore_errors=True)
    os.mkdir(f'results/{loc}')
    os.mkdir(f'results/{loc}/partial')
    os.mkdir(f'results/{loc}/plots')

    # Run agents
    thunks = [(env, agent, i * args.reps + j, args, (seed, random.randint(0, (1 << 32) - 1)), loc)
                            for i, agent in enumerate(args.agents)
                            for j in range(args.reps)]
    random.shuffle(thunks)
    pool = multiprocessing.Pool(processes=args.cpus, maxtasksperchild=1)
    results = [pool.apply_async(run_agent, [thunk]) for thunk in thunks]
    end_time = time.time() + args.timeout
    collected = [x for x in [get_result(result, end_time - time.time()) for result in results] if x is not None]

    # Write output
    np.save(f'results/{loc}/results.npy', collected)

    for metric in args.metrics:
        process_data(metric, collected)


import os
import sys
import json
import signal
import argparse
import subprocess
import multiprocessing

signal.signal(signal.SIGINT, lambda x, y: exit(1))
dir_path = os.path.dirname(os.path.realpath(__file__))
parser = argparse.ArgumentParser()
parser.add_argument('file', type=str, help='JSON file with jobs to run')
parser.add_argument('-n', type=int, default=None, help='number of jobs to run simultaneously')
args = parser.parse_args()
jobs = json.loads(open(args.file).read())

def run_job(job):
    print(f'Running job: {job}')

    args = [sys.executable, f'{dir_path}/run.py']
    
    def val_arg(arg):
        nonlocal args
        if arg in job:
            args += [f'--{arg}', str(job[arg])]

    def bool_arg(arg):
        nonlocal args
        if arg in job and job[arg]:
            args += [f'--{arg}']

    def multi_arg(arg):
        nonlocal args
        if arg in job:
            args += [f'--{arg}', *map(str, job[arg])]

    multi_arg('agents')

    for arg in ['nocorr', 'pretrain']:
        bool_arg(arg)

    for arg in ['batch', 'cutoff', 'validation', 'env', 'reps', 'name', 'cpus']:
        val_arg(arg)

    try: os.mkdir(f'results/{job["name"]}')
    except OSError: pass
    subprocess.run(args, stderr=open(f'results/{job["name"]}/err.log','w'), stdout=open(f'results/{job["name"]}/out.log', 'w'))

pool = multiprocessing.Pool(processes=args.n, maxtasksperchild=1)
pool.map(run_job, jobs, chunksize=1)


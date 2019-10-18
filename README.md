# Setup
`pip install -r requirements.txt`

# Usage
Any agent in the agents directory can be tested:

`python run.py --agents [Agent1 Agent2 ...] --batch [batchsize]`

e.g. `python run.py --agents 'RandomAgent(epochs=10)' 'GreedyAgent(epochs=10)' --batch 1000`

# Flags

`--env 'GenericEnv("data/toy/20mer.csv")'`: use X, Y data in provided file.

`--reps [N]`: average multiple trials.

`--cutoff [N]`: limit to N batches.

`--pretrain`: use pretraining data.

`--nocorr`: compute no prediction correlations.

# Gym
Install OpenAI gym:

`cd gym_batgirl`

`pip install -e .`

To use:

`env = gym.make('batgirl-v0')`

`actions = env.reset('protein-BRCA1', batch=100)`

`obs, regret, done, info = env.step(actions[:100])`



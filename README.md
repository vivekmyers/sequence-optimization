# Setup
`pip install -r requirements.txt`

# Usage
Any agent in the agents directory can be tested:
`python run.py --agents [Agent1 Agent2 ...] --batch [batchsize]`

e.g. `python run.py --agents BaseAgent 'GreedyAgent(epochs=10)' --batch 1000 --collect`

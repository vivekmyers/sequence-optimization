import numpy as np
from random import *
from models.uncertain import UncertainCNN
from models.auto_cnn import CNN
import agents.random
from torch.distributions import Normal
from torch import tensor


def PseudoThompsonAgent(epochs=30):
    '''Constructs agent with a CNN trained to predict gaussians with uncertainty, 
    using Thompson sampling with the network's uncertainty to select batches, and 
    fitting the model to update the predicted distributions between batches.
    '''

    class Agent(agents.random.RandomAgent(epochs)):

        def __init__(self, *args):
            super().__init__(*args)
            self.model = UncertainCNN(encoder=self.encode, shape=self.shape)
            
        def act(self, seqs):
            mu, sigma = map(tensor, self.model.predict(seqs))
            sampled = Normal(mu, sigma).sample().numpy()
            return [*zip(*sorted(zip(sampled, seqs))[-self.batch:])][1]

        def observe(self, data):
            super().observe(data)
            self.model.fit(*zip(*self.seen.items()), epochs=epochs)
        
    return Agent

import numpy as np
from random import *
from models.bayesian import BayesianCNN
from models.cnn import CNN
import agents.base

def ThompsonAgent(epochs=10, initial_epochs=None):
    '''Constructs agent with a Bayesian CNN, using Thompson sampling with the
    network's uncertainty (over its parameters) to select batches, and 
    refitting the model to update the predicted distributions between batches.
    '''
    if initial_epochs is None:
        initial_epochs = epochs // 4

    class Agent(agents.base.BaseAgent):

        def __init__(self, *args):
            super().__init__(*args)
            self.model = BayesianCNN(encoder=self.encode, shape=self.shape)
            if len(self.prior):
                self.model.fit(*zip(*self.prior.items()), epochs=initial_epochs, minibatch=100)
        
        def act(self, seqs):
            mu, sigma = self.model.sample(seqs)
            return list(zip(*sorted(zip(mu, seqs))[-self.batch:]))[1]

        def observe(self, data):
            super().observe(data)
            self.model.fit(*zip(*self.seen.items()), epochs=epochs, minibatch=min(len(self.seen), 100))
        
        def predict(self, seqs):
            result = np.zeros([len(seqs)])
            while not result.std():
                model = CNN(encoder=self.encode, shape=self.shape)
                if self.prior: model.fit(*zip(*self.prior.items()), epochs=initial_epochs, minibatch=100)
                if self.seen: model.fit(*zip(*self.seen.items()), epochs=epochs, minibatch=100)
                result = model.predict(seqs)
            return result

    return Agent


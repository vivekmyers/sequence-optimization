import numpy as np
from random import *
import agents.base
from models.gp import GaussianProcess
from models.cnn import CNN

def GaussianAgent(epochs=10, initial_epochs=None, dim=5, tau=0.01, beta=0.2):
    '''Constructs agent that uses batch version of GP-UCB algorithm to sample
    sequences with a deep kernel gaussian process regression.
    dim: embedding dimension.
    tau: kernel covariance parameter.
    beta: relative weight of sequence score in generating embedding.
    '''
    if initial_epochs is None:
        initial_epochs = epochs // 4

    class Agent(agents.base.BaseAgent):

        def __init__(self, *args):
            super().__init__(*args)
            self.model = GaussianProcess(encoder=self.encode, dim=dim, shape=self.shape, tau=tau, beta=beta)
            if len(self.prior):
                self.model.embed.refit(*zip(*self.prior.items()), epochs=initial_epochs, 
                                        minibatch=100)
        
        def act(self, seqs):
            prior = {}
            choices = []
            t = 1 + len(self.seen) // self.batch
            D = len(seqs) + len(self.seen)
            beta = lambda t: 2 * np.log(D * t ** 2 * np.pi ** 2 / 3)
            mu = self.model.interpolate(seqs, prior)
            sigma = self.model.uncertainty(seqs, prior)
            yt = (mu - np.sqrt(beta(t)) * sigma).max()
            seqs = np.array(seqs)
            idx = np.argmax(mu + np.sqrt(beta(t)) * sigma)
            x0 = seqs[idx]
            Rt = seqs[mu + 2 * np.sqrt(beta(t + 1)) * sigma >= yt]
            Rt = np.array([x for x in Rt if x != x0])
            prior[x0] = None
            choices = [x0]
            for i in range(self.batch - 1):
                if len(Rt) == 0:
                    choices.append(choice([x for x in seqs if x not in choices]))
                    continue
                sigma = self.model.uncertainty(Rt, prior)
                idx = np.argmax(sigma)
                xk = Rt[idx]
                Rt = np.delete(Rt, idx, 0)
                prior[xk] = None
                choices.append(xk)
            return choices

        def observe(self, data):
            super().observe(data)
            self.model.fit(*zip(*self.seen.items()), epochs=epochs, 
                                minibatch=min(len(self.seen), 100))
        
        def predict(self, seqs):
            result = np.zeros([len(seqs)])
            while not result.std():
                model = CNN(encoder=self.encode, shape=self.shape)
                if self.prior: model.fit(*zip(*self.prior.items()), epochs=initial_epochs, minibatch=100)
                if self.seen: model.fit(*zip(*self.seen.items()), epochs=epochs, minibatch=100)
                result = model.predict(seqs)
            return result

    return Agent


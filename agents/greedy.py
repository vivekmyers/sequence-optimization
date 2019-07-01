import numpy as np
from random import *
from models.cnn import CNN
import agents.base

class GreedyAgent(agents.base.BaseAgent):
    def __init__(self, *args):
        super().__init__(*args)
        self.model = CNN(shape=[self.len, 4])
    
    def act(self, seqs):
        return list(zip(*sorted(zip(self.predict(seqs), seqs))[-self.batch:]))[1]

    def observe(self, data):
        super().observe(data)
        self.model.fit(*zip(*self.seen.items()), epochs=10, verbose=0)
    
    def predict(self, seqs):
        return self.model.predict(seqs)

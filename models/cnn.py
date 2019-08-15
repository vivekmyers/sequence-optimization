import numpy as np
from random import *
import os, sys
import torch
from torch import nn
import torch.functional as F
import utils.model

class CNN:
    '''CNN with regularization for making simple sequence score predictions.'''

    def _make_net(self, alpha, opt, shape):

        class Model(nn.Module):

            def __init__(self):
                super().__init__()
                conv = self.conv = [nn.Conv1d(shape[1], 64, 7, stride=1, padding=3),
                    nn.Conv1d(64, 64, 5, stride=1, padding=2),
                    nn.Conv1d(64, 32, 3, stride=1, padding=1)]
                self.conv_layers = nn.Sequential(
                    conv[0], nn.ReLU(), conv[1], nn.ReLU(),
                    conv[2], nn.ReLU()) 
                fc = self.fc = [nn.Linear(32 * shape[0], 100), nn.Linear(100, 100), nn.Linear(100, 1)]
                self.fc_layers = nn.Sequential(
                    fc[0], nn.ReLU(), nn.Dropout(0.5), fc[1], nn.ReLU(), 
                    nn.Dropout(0.5), fc[2], nn.Sigmoid())
            
            def forward(self, x):
                filtered = self.conv_layers(x.permute(0, 2, 1))
                return torch.squeeze(self.fc_layers(filtered.reshape(filtered.shape[0], -1)), dim=1)
            
            def l2(self):
                return sum(torch.sum(param ** 2) for c in self.conv for param in c.parameters())

        self.model = Model().to(self.device)

    def fit(self, seqs, scores, epochs):
        self.model.train()
        D = list(zip([self.encode(x) for x in seqs], scores))
        M = len(D) // self.minibatch + bool(len(D) % self.minibatch)
        for ep in range(epochs):
            shuffle(D)
            for mb in range(M):
                X, Y = map(lambda t: torch.tensor(t).to(self.device).float(), 
                                zip(*D[mb * self.minibatch : (mb + 1) * self.minibatch]))
                loss = torch.sum((Y - self.model(X)) ** 2) + self.lam * self.model.l2()
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                self.opt.step()
    
    @utils.model.batch
    def predict(self, seqs):
        self.model.eval()
        return self.model(torch.tensor(
            [self.encode(seq) for seq in seqs]).to(self.device).float()).detach().cpu().numpy()
    
    def __call__(self, seqs):
        return self.predict(seqs)

    def __init__(self, encoder, alpha=5e-4, shape=(), lam=1e-3, minibatch=100):
        '''encoder: convert sequences to one-hot arrays.
        alpha: learning rate.
        shape: sequence shape.
        lam: l2 regularization constant.
        '''
        super().__init__()
        if not torch.cuda.is_available():
            self.device = 'cpu'
        else:
            self.device = 'cuda'
        self.minibatch = minibatch
        self.encode = encoder
        self.lam = lam
        self.alpha = alpha
        self._make_net(alpha, 'adam', shape)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.alpha)


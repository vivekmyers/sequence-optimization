import numpy as np
from random import *
import os, sys
import torch
from torch import nn
import torch.functional as F
import utils.model


class Featurizer:
    '''Learns predictive model over sequences, then trains autoencoder
    on convolutional output to create embedding.
    '''

    def _make_net(self, alpha, shape, dim, hidden=100):

        class Featurizer(nn.Module):

            def __init__(self):
                super().__init__()
                conv = self.conv = [nn.Conv1d(shape[1], 64, 7, stride=1, padding=3),
                    nn.Conv1d(64, 64, 5, stride=1, padding=2),
                    nn.Conv1d(64, 32, 3, stride=1, padding=1)]
                self.conv_layers = nn.Sequential(
                    conv[0], nn.ReLU(), conv[1], nn.ReLU(),
                    conv[2], nn.ReLU()) 

            def forward(self, x):
                filtered = self.conv_layers(x.permute(0, 2, 1))
                return filtered.reshape(filtered.shape[0], -1)

            def l2(self):
                return sum(torch.sum(param ** 2) for c in self.conv for param in c.parameters())


        class Predictor(nn.Module):

            def __init__(self):
                super().__init__()
                fc = self.fc = [nn.Linear(32 * shape[0], hidden), nn.Linear(hidden, 1)] 
                self.fc_layers = nn.Sequential(fc[0], nn.ReLU(), fc[1], nn.Sigmoid()) 
            
            def forward(self, x):
                return self.fc_layers(x).squeeze(dim=1)

            def l2(self):
                return sum(torch.sum(param ** 2) for c in self.fc for param in c.parameters())


        class Encoder(nn.Module):

            def __init__(self):
                super().__init__()
                fc = self.fc = [nn.Linear(32 * shape[0], hidden), nn.Linear(hidden, dim)]
                self.fc_layers = nn.Sequential(
                    fc[0], nn.ReLU(), fc[1], nn.Sigmoid())
            
            def forward(self, x):
                return self.fc_layers(x)
            
            def l2(self):
                return sum(torch.sum(param ** 2) for c in self.fc for param in c.parameters())

        class Decoder(nn.Module):

            def __init__(self):
                super().__init__()
                fc = self.fc = [nn.Linear(dim, hidden), nn.Linear(hidden, 32 * shape[0])] 
                self.fc_layers = nn.Sequential(fc[0], nn.ReLU(), fc[1], nn.ReLU()) 
            
            def forward(self, x):
                return self.fc_layers(x)

            def l2(self):
                return sum(torch.sum(param ** 2) for c in self.fc for param in c.parameters())

        self.featurizer = Featurizer().to(self.device)
        self.predictor = Predictor().to(self.device)
        self.encoder = Encoder().to(self.device)
        self.decoder = Decoder().to(self.device)
        self.params = [*self.featurizer.parameters(), *self.predictor.parameters(),
                       *self.encoder.parameters(), *self.decoder.parameters()]

    def fit(self, seqs, scores, epochs):
        D = [(self.encode(x), y) for x, y in zip(seqs, scores)]
        M = len(D) // self.minibatch + bool(len(D) % self.minibatch)
        for ep in range(epochs):
            shuffle(D)
            for mb in range(M):
                X, Y = [torch.tensor(t).to(self.device).float()
                        for t in zip(*D[mb * self.minibatch : (mb + 1) * self.minibatch])]
                F = self.featurizer(X)
                F_hat = self.decoder(self.encoder(F.detach()))
                Y_hat = self.predictor(F)
                F_loss = (F - F_hat).pow(2).mean(dim=1).sum() 
                Y_loss = (Y - Y_hat).pow(2).sum()
                l2 = self.lam * (self.encoder.l2() + self.decoder.l2() \
                        + self.featurizer.l2() + self.predictor.l2())
                loss = F_loss + Y_loss + l2
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.params, 1)
                self.opt.step()

    @utils.model.batch
    def predict(self, seqs):
        '''Predict scores.'''
        D = torch.tensor([self.encode(x) for x in seqs]).to(self.device).float()
        Y_hat = self.predictor(self.featurizer(D))
        return Y_hat.cpu().detach().numpy()
    
    @utils.model.batch
    def embed(self, seqs):
        '''Encode list of sequences.'''
        D = torch.tensor([self.encode(x) for x in seqs]).to(self.device).float()
        em = self.encoder(self.featurizer(D))
        return em.cpu().detach().numpy()

    def __call__(self, seqs):
        return self.embed(seqs)

    def __init__(self, encoder, shape, dim=5, alpha=5e-4, lam=0., minibatch=100):
        '''encoder: convert sequences to one-hot arrays.
        dim: dimensionality of embedding.
        alpha: learning rate.
        shape: sequence shape.
        lam: l2 regularization constant.
        minibatch: minibatch size
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
        self._make_net(alpha, shape, dim)
        self.opt = torch.optim.Adam(self.params, lr=self.alpha)


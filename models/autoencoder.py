import numpy as np
from random import *
import os, sys
import torch
from torch import nn
import torch.functional as F

class Autoencoder:
    '''Learns encoder and decoder for unsupervised sequence embedding.'''

    def _make_net(self, alpha, shape, dim):

        class Encoder(nn.Module):

            def __init__(self):
                super().__init__()
                conv = self.conv = [nn.Conv1d(shape[1], 64, 7, stride=1, padding=3),
                    nn.Conv1d(64, 64, 5, stride=1, padding=2),
                    nn.Conv1d(64, 32, 3, stride=1, padding=1)]
                self.conv_layers = nn.Sequential(
                    conv[0], nn.ReLU(), conv[1], nn.ReLU(),
                    conv[2], nn.ReLU()) 
                fc = self.fc = [nn.Linear(32 * shape[0], 200), nn.Linear(200, dim)]
                self.fc_layers = nn.Sequential(
                    fc[0], nn.ReLU(), fc[1], nn.Sigmoid())
            
            def forward(self, x):
                filtered = self.conv_layers(x.permute(0, 2, 1))
                return self.fc_layers(filtered.reshape(filtered.shape[0], -1))
            
            def l2(self):
                return sum(torch.sum(param ** 2) for c in self.conv + self.fc for param in c.parameters())

        class Decoder(nn.Module):

            def __init__(self):
                super().__init__()
                fc = self.fc = [nn.Linear(dim, 100), nn.Linear(100, 200), nn.Linear(200, shape[0] * shape[1] + 1)] 
                self.fc_layers = nn.Sequential(fc[0], nn.ReLU(), fc[1], nn.ReLU(), fc[2]) 
            
            def forward(self, x):
                result = self.fc_layers(x)
                return result[:, : -1], result[:, -1]

            def l2(self):
                return sum(torch.sum(param ** 2) for c in self.fc for param in c.parameters())

        self.encoder = Encoder().to(self.device)
        self.decoder = Decoder().to(self.device)

    def refit(self, seqs, scores, epochs, minibatch):
        D = [(self.encode(x), y) for x, y in zip(seqs, scores)]
        M = len(D) // minibatch
        for ep in range(epochs):
            shuffle(D)
            for mb in range(M):
                X, Y = [torch.tensor(t).to(self.device).float() for t in zip(*D[mb * minibatch : (mb + 1) * minibatch])]
                X_hat, Y_hat = self.decoder(self.encoder(X))
                l2 = self.lam * (self.encoder.l2() + self.decoder.l2())
                LX = torch.sum((X_hat - X.view(X_hat.shape)) ** 2) / X_hat.shape[1]
                LY = torch.sum((Y_hat - Y) ** 2)
                loss = LX * (1 - self.beta) + LY * self.beta + l2
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                        [*self.encoder.parameters(), 
                            *self.decoder.parameters()], 1)
                self.opt.step()

    def predict(self, seqs):
        '''Predict scores using decoder.'''
        D = torch.tensor([self.encode(x) for x in seqs]).to(self.device).float()
        X, Y = self.decoder(self.encoder(D))
        return Y.cpu().detach().numpy()
    
    def __call__(self, seqs):
        '''Encode list of sequences.'''
        D = torch.tensor([self.encode(x) for x in seqs]).to(self.device).float()
        return self.encoder(D).cpu().detach().numpy()

    def __init__(self, encoder, shape, dim=5, beta=0., alpha=5e-4, lam=1e-3):
        '''encoder: convert sequences to one-hot arrays.
        dim: dimensionality of embedding.
        alpha: learning rate.
        shape: sequence shape.
        lam: l2 regularization constant.
        beta: weighting for Y prediction
                (0 for normal autoencoder, 1 for predictive embedding).
        '''
        super().__init__()
        if not torch.cuda.is_available():
            self.device = 'cpu'
        else:
            self.device = 'cuda'
        self.encode = encoder
        self.lam = lam
        self.alpha = alpha
        self.beta = beta
        self._make_net(alpha, shape, dim)
        self.opt = torch.optim.Adam(
                [*self.encoder.parameters(), *self.decoder.parameters()], 
                lr=self.alpha)


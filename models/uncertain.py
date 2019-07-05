import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal
from random import shuffle
import numpy as np


class SeqConv(nn.Module):
    '''Three 1D convolutions followed by two dense layers, operating on sequence data.'''

    def __init__(self, length, channels):
        super().__init__()
        self.length = length
        mk = lambda *x: nn.Parameter(torch.empty(*x, requires_grad=True))
        self.W1_conv = mk(64, channels, 7)
        self.B1_conv = mk(64)
        self.W2_conv = mk(64, 64, 5)
        self.B2_conv = mk(64)
        self.W3_conv = mk(32, 64, 3)
        self.B3_conv = mk(32)
        self.W1_fc = mk(32 * length, 100)
        self.B1_fc = mk(100)
        self.W2_fc = mk(100, 2)
        self.B2_fc = mk(2)

    def forward(self, x):
        x = x.permute(0, 2, 1).to(dtype=torch.float)
        x = F.relu(F.conv1d(x, self.W1_conv, self.B1_conv, padding=3))
        x = F.relu(F.conv1d(x, self.W2_conv, self.B2_conv, padding=2))
        x = F.relu(F.conv1d(x, self.W3_conv, self.B3_conv, padding=1))
        x = x.reshape(x.shape[0], -1)
        x = F.relu(x @ self.W1_fc + self.B1_fc)
        x = x @ self.W2_fc  + self.B2_fc
        return x
    

class UncertainCNN:
    '''Neural network that predicts gaussians around its outputs.'''
    
    def _make_net(self, shape):
        self.model = SeqConv(*shape).to(self.device)
        self.params = list(self.model.parameters()) # keep model parameters in list of tensors
        for param in self.params:
            if len(param.shape) > 1:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.normal_(param)

    def fit(self, seqs, scores, epochs, minibatch):
        '''Fit encoded sequences to provided scores for provided epochs,
        sampling a model for each minibatch of the provided size.
        '''
        D = [(self.encode(x), y) for x, y in zip(seqs, scores)] # data tuples
        M = len(D) // minibatch # number of minibatches

        for ep in range(epochs):
            shuffle(D)
            for mb in range(M):

                # get minibatch of X values, and predicted (mu, sigma) for each Y 
                Di = D[mb * minibatch : (mb + 1) * minibatch]
                X = self._process([x for x, y in Di])
                Y = torch.tensor([y for x, y in Di]).to(self.device)
                pred = self.model(X)
                Y_mu, Y_rho = torch.sigmoid(pred[:, 0]), torch.log(1 + torch.exp(pred[:, 1]))

                # loss function
                p_w = sum(Normal(0, 1).log_prob(weight).sum() for weight in self.params) # weights prior
                p_D = Normal(Y_mu, Y_rho + self._eps).log_prob(Y).sum() # prediction loss
                loss = - (p_w + p_D)

                # compute and apply gradients
                self.opt.zero_grad()
                loss.backward()
                for weight in self.params:
                    # we clip gradients to avoid exploding logprobs
                    nn.utils.clip_grad_norm_(weight, 1)
                self.opt.step()              
                    
    def predict(self, seqs):
        '''Return (mus, sigmas) for the sequences describing a gaussian for the predicted
        scores of each one.
        '''
        result = self.model(self._process([self.encode(x) for x in seqs]))
        return torch.sigmoid(result[:, 0]).detach().numpy(), torch.exp(result[:, 1]).add(1).log().detach().numpy()
    
    def __call__(self, seqs):
        return self.predict(seqs)

    def __init__(self, encoder, alpha=1e-4, shape=()):
        '''Takes sequence encoding function, step size, and sequence shape.'''
        super().__init__()
        if not torch.cuda.is_available(): 
            print('CUDA not available')
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        self.alpha = alpha
        self._process = lambda x: torch.tensor(np.array(x), requires_grad=True).to(self.device)
        self._make_net(shape)
        self._eps = 1e-6
        self.encode = encoder
        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.alpha)


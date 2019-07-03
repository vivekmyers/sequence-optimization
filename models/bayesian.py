import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal
from random import shuffle
import numpy as np

class SeqConv(nn.Module):
    '''
    Three 1D convolutions followed by two dense layers, operating on sequence data. 
    '''
    
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
        x = x.view(x.shape[0], x.shape[1] * x.shape[2])
        x = F.relu(x @ self.W1_fc + self.B1_fc)
        x = x @ self.W2_fc  + self.B2_fc
        return x

class BayesianCNN:
    '''
    Bayesian neural network. Predicts uncertainty and keeps uncertainty for each
    model parameter.
    '''
    
    def make_net(self, shape):
        self.model = SeqConv(*shape).to(self.device)
        self.params = list(self.model.parameters())
        self.mu = [torch.empty(*x.shape, requires_grad=True).to(self.device) for x in self.params]
        self.sigma = [torch.full(x.shape, 1, requires_grad=True).to(self.device) for x in self.params]
        for var in self.mu:
            nn.init.normal_(var)
        for x, y in zip(self.params, self.mu):
            x.data.copy_(y)
            
    def dist(self):
        '''
        Returns list of model parameter distribution gaussians.
        '''
        return [Normal(mu, torch.log(1 + torch.exp(sigma))) 
                for mu, sigma in zip(self.mu, self.sigma)]

    def fit(self, seqs, scores, epochs, minibatch):
        D = [(self.encode(x), y) for x, y in zip(seqs, scores)]
        M = len(D) // minibatch
        pD = [Normal(torch.zeros(*x.shape).to(self.device), 
                torch.full(x.shape, 1).to(self.device)) for x in self.params]
        for ep in range(epochs):
            shuffle(D)
            dist = self.dist()
            eps = [torch.normal(torch.zeros(*x.shape), torch.full(x.shape, 1)) for x in self.params]
            w = [mu + d * (1 + sigma.exp()).log() for mu, sigma, d in zip(self.mu, self.sigma, eps)]
            for i, weight in enumerate(w):
                self.params[i].data = weight
                weight.retain_grad()
            for mb in range(M):
                Di = D[mb * minibatch : (mb + 1) * minibatch]
                X = self.process([x for x, y in Di])
                Y = torch.tensor([y for x, y in Di], requires_grad=True).to(self.device)
                pred = self.model(X)
                Y_mu, Y_sigma = pred[:, 0], torch.log(1 + torch.exp(pred[:, 1]))
                F = 1 / M * (
                    sum(n.log_prob(weight).sum() for weight, n in zip(w, dist)) \
                    - sum(n.log_prob(weight).sum() for weight, n in zip(w, pD))) \
                    - Normal(Y_mu, Y_sigma).log_prob(Y).sum()
                for var in self.mu + self.sigma + self.params:
                    if var.grad is not None: var.grad.data.zero_()
                F.backward(retain_graph=mb < M - 1)
                del_mu = [weight.grad + mu.grad for weight, mu in zip(w, self.mu)]
                del_sig = [torch.div(d * weight.grad + sigma.grad, 1 + torch.exp(-sigma)) 
                                for d, weight, sigma in zip(eps, w, self.sigma)]
                for mu, dmu in zip(self.mu, del_mu):
                    mu.data.copy_(mu - self.alpha * dmu)
                for sigma, dsigma in zip(self.sigma, del_sig):
                    sigma.data.copy_(sigma - self.alpha * dsigma)              
                    
    def predict(self, seqs):
        '''
        Return (mus, sigmas) for the sequences describing a gaussian for the predicted
        scores of each one.
        '''
        
        for param, mu in zip(self.params, self.mu):
            param.data.copy_(mu)
        result = self.model(self.process([self.encode(x) for x in seqs]))
        return result[:, 0].detach().numpy(), torch.log(1 + torch.exp(result[:, 1])).detach().numpy()
    
    def sample(self, seqs):
        '''
        Sample a model theta from the model distribution conditioned on all observed data,
        then return the (mus, sigmas) predicted by theta.
        '''
        
        for param, weight in zip(self.params, [n.sample() for n in self.dist()]):
            param.data.copy_(weight)
        result = self.model(self.process([self.encode(x) for x in seqs]))
        return result[:, 0].detach().numpy(), torch.log(1 + torch.exp(result[:, 1])).detach().numpy()
    
    def __call__(self, seqs):
        return self.predict(seqs)

    def __init__(self, encoder, alpha=1e-4, shape=()):
        super().__init__()
        if not torch.cuda.is_available(): 
            print('CUDA not available')
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        self.alpha = alpha
        self.process = lambda x: torch.tensor(np.array(x), requires_grad=True).to(self.device)
        self.make_net(shape)
        self.encode = encoder

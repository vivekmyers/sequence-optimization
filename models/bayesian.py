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


class BayesianCNN:
    '''Bayesian neural network. Predicts uncertainty and keeps uncertainty for each
    model parameter.
    '''
    
    def make_net(self, shape):
        self.model = SeqConv(*shape).to(self.device)
        self.params = list(self.model.parameters()) # keep model parameters in list of tensors
        self.mu = [torch.empty(*x.shape, requires_grad=True).to(self.device) 
                for x in self.params] # mean model parameters
        for var in self.mu:
            if len(var.shape) > 1:
                nn.init.xavier_uniform_(var) # weights
            else:
                nn.init.normal_(var) # biases
        self.rho = [torch.full(x.shape, x.detach().std().exp().add(-1).log(), 
                        requires_grad=True).to(self.device)
                        for x in self.mu] # scaled stdevs of model parameters
        for x, y in zip(self.params, self.mu):
            x.data.copy_(y)
            
    def dist(self):
        '''Returns list of model parameter distribution gaussians.
        '''
        return [Normal(mu, torch.log(1 + torch.exp(rho))) 
                for mu, rho in zip(self.mu, self.rho)]

    def fit(self, seqs, scores, epochs, minibatch):
        '''Fit encoded sequences to provided scores for provided epochs,
        sampling a model for each minibatch of the provided size.
        '''
        D = [(self.encode(x), y) for x, y in zip(seqs, scores)] # data tuples
        M = len(D) // minibatch # number of minibatches
        pD = [Normal(torch.zeros(*x.shape).to(self.device), # prior unit gaussion over weights
                torch.full(x.shape, 1).to(self.device)) for x in self.params]

        for ep in range(epochs):
            shuffle(D)
            for mb in range(M):

                # sample model weights from N(self.mu, self.rho)
                eps = [torch.normal(torch.zeros(*x.shape), torch.full(x.shape, 1)) for x in self.params]
                w = [mu + d * (1 + rho.exp()).log() for mu, rho, d in zip(self.mu, self.rho, eps)]
                for param, weight in zip(self.params, w):
                    param.data.copy_(weight)

                # get minibatch of X values, and predicted (mu, sigma) for each Y 
                Di = D[mb * minibatch : (mb + 1) * minibatch]
                X = self.process([x for x, y in Di])
                Y = torch.tensor([y for x, y in Di], requires_grad=True).to(self.device)
                pred = self.model(X)
                Y_mu, Y_rho = pred[:, 0], torch.log(1 + torch.exp(pred[:, 1]))

                # loss function
                q_w = sum(n.log_prob(weight).sum() 
                        for weight, n in zip(w, self.dist())) # variational posterior
                p_w = sum(n.log_prob(weight).sum() for weight, n in zip(w, pD)) # weights prior
                p_D = Normal(Y_mu, Y_rho + self.alpha).log_prob(Y).sum() # prediction loss
                loss = (q_w - p_w) / M - p_D

                # compute and apply gradients
                for var in self.mu + self.rho + self.params + w:
                    if var.grad is not None: var.grad.data.zero_()
                loss.backward(retain_graph=mb < M - 1)
                for weight in self.params:
                    # we clip gradients to avoid exploding logprobs
                    nn.utils.clip_grad_norm_(weight, 1)
                del_mu = [weight.grad + mu.grad for weight, mu in zip(self.params, self.mu)]
                del_rho = [torch.div(d * weight.grad, 1 + torch.exp(-rho)) + rho.grad  
                                for d, weight, rho in zip(eps, self.params, self.rho)]

                # update self.mu, self.rho
                for mu, dmu in zip(self.mu, del_mu):
                    mu.data.copy_(mu - self.alpha * dmu)
                for rho, drho in zip(self.rho, del_rho):
                    rho.data.copy_(rho - self.alpha * drho)              
                    
    def predict(self, seqs):
        '''Return (mus, sigmas) for the sequences describing a gaussian for the predicted
        scores of each one.
        '''
        for param, mu in zip(self.params, self.mu):
            param.data.copy_(mu)
        result = self.model(self.process([self.encode(x) for x in seqs]))
        return result[:, 0].detach().numpy(), torch.log(1 + torch.exp(result[:, 1])).detach().numpy()
    
    def sample(self, seqs):
        '''Sample a model theta from the model distribution conditioned on all observed data,
        then return the (mus, sigmas) predicted by theta.
        '''
        for param, weight in zip(self.params, [n.sample() for n in self.dist()]):
            param.data.copy_(weight)
        result = self.model(self.process([self.encode(x) for x in seqs]))
        return result[:, 0].detach().numpy(), torch.log(1 + torch.exp(result[:, 1])).detach().numpy()
    
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
        self.process = lambda x: torch.tensor(np.array(x), requires_grad=True).to(self.device)
        self.make_net(shape)
        self.encode = encoder

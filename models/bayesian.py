import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal
from random import shuffle
import numpy as np


class BayesianCNN:
    '''Bayesian neural network. Predicts uncertainty and keeps uncertainty for each
    model parameter.
    '''
    
    def _seq_conv(self, length, channels): # get list of empty model parameters
        mk = lambda *x: torch.empty(*x, requires_grad=True).to(self.device)
        W1_cv = mk(64, channels, 7)
        B1_cv = mk(64)
        W2_cv = mk(64, 64, 5)
        B2_cv = mk(64)
        W3_cv = mk(32, 64, 3)
        B3_cv = mk(32)
        W1_fc = mk(32 * length, 100)
        B1_fc = mk(100)
        W2_fc = mk(100, 2)
        B2_fc = mk(2)
        return [W1_cv, B1_cv, W2_cv, B2_cv, W3_cv, B3_cv, W1_fc, B1_fc, W2_fc, B2_fc]
                    
    def _model(self, w, x): # apply parameters w to input x
        x = x.permute(0, 2, 1).to(dtype=torch.float)
        x = F.relu(F.conv1d(x, w[0], w[1], padding=3))
        x = F.relu(F.conv1d(x, w[2], w[3], padding=2))
        x = F.relu(F.conv1d(x, w[4], w[5], padding=1))
        x = x.reshape(x.shape[0], -1)
        x = F.relu(x @ w[6] + w[7])
        x = x @ w[8] + w[9]
        return x

    def _make_net(self, shape, sig_scale):
        self.mu = self._seq_conv(*shape) # mean model parameters
        for var in self.mu:
            if len(var.shape) > 1:
                nn.init.xavier_uniform_(var) # weights
            else:
                nn.init.normal_(var) # biases
        self.rho = [torch.full(x.shape, x.detach().std().mul(sig_scale).exp().add(-1).log(), 
                        requires_grad=True).to(self.device)
                        for x in self.mu] # scaled stdevs of model parameters
            
    def dist(self):
        '''Returns list of model parameter distribution gaussians.
        '''
        return [Normal(mu, rho.exp().add(1).log() + self._eps) for mu, rho in zip(self.mu, self.rho)]

    def fit(self, seqs, scores, epochs, minibatch):
        '''Fit encoded sequences to provided scores for provided epochs,
        sampling a model for each minibatch of the provided size.
        '''
        D = [(self.encode(x), y) for x, y in zip(seqs, scores)] # data tuples
        M = len(D) // minibatch # number of minibatches

        for ep in range(epochs):
            shuffle(D)
            for mb in range(M):

                # sample model weights from N(self.mu, self.rho)
                eps = [torch.normal(torch.zeros(*x.shape), torch.full(x.shape, 1) + self._eps) for x in self.mu]
                w = [mu + d * (1 + rho.exp()).log() for mu, rho, d in zip(self.mu, self.rho, eps)]

                # get minibatch of X values, and predicted (mu, sigma) for each Y 
                Di = D[mb * minibatch : (mb + 1) * minibatch]
                X = self._process([x for x, y in Di])
                Y = torch.tensor([y for x, y in Di]).to(self.device)
                pred = self._model(w, X)
                Y_mu, Y_rho = torch.sigmoid(pred[:, 0]), torch.log(1 + torch.exp(pred[:, 1]))

                # loss function
                q_w = sum(n.log_prob(weight).sum() 
                        for weight, n in zip(w, self.dist())) # variational posterior
                p_w = sum(Normal(0, 1).log_prob(weight).sum() for weight in w) # weights prior
                p_D = Normal(Y_mu, Y_rho + self._eps).log_prob(Y).sum() # prediction loss
                loss = (q_w - p_w) / M - p_D

                # compute and apply gradients
                if torch.isnan(loss):
                    continue
                self.opt.zero_grad()
                loss.backward(retain_graph=mb < M - 1)
                for weight in self.mu + self.rho:
                    # we clip gradients to avoid exploding logprobs
                    nn.utils.clip_grad_norm_(weight, 1)
                self.opt.step()
                    
    def predict(self, seqs):
        '''Return (mus, sigmas) for the sequences describing a gaussian for the predicted
        scores of each one.
        '''
        X = self._process([self.encode(x) for x in seqs])
        result = self._model(self.mu, X)
        return torch.sigmoid(result[:, 0]).detach().numpy(), result[:, 1].exp().add(1).log().detach().numpy()
    
    def sample(self, seqs):
        '''Sample a model theta from the model distribution conditioned on all observed data,
        then return the (mus, sigmas) predicted by theta.
        '''
        X = self._process([self.encode(x) for x in seqs])
        w = [n.sample() for n in self.dist()]
        result = self._model(w, X)
        return torch.sigmoid(result[:, 0]).detach().numpy(), result[:, 1].exp().add(1).log().detach().numpy()
    
    def __call__(self, seqs):
        return self.predict(seqs)

    def __init__(self, encoder, alpha=1e-4, shape=(), sig_scale=0.5):
        '''Takes sequence encoding function, step size, sequence shape, and scaling 
        of initial weight stdevs.
        '''
        super().__init__()
        if not torch.cuda.is_available(): 
            print('CUDA not available')
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        self.alpha = alpha
        self.encode = encoder
        self._eps = 1e-6
        self._process = lambda x: torch.tensor(np.array(x), requires_grad=True).to(self.device)
        self._make_net(shape, sig_scale)
        self.opt = torch.optim.Adam(self.mu + self.rho, lr=self.alpha)


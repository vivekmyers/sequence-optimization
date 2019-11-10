import gpytorch
import torch

class ExactGPModel(gpytorch.models.ExactGP):
    '''Exact GP model.'''

    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class FittedGP:
    '''GP on fixed X, Y data which can have hyperparameters tuned.'''

    def __init__(self, X, Y):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.X, self.Y = map(lambda x: torch.tensor(x).to(self.device).float(), [X, Y])
        self.model = ExactGPModel(self.X, self.Y, gpytorch.likelihoods.GaussianLikelihood()).to(self.device).float()
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
        self.likelihood.train()
        self.optim = torch.optim.Adam(self.model.parameters(), lr=0.1)
        
    def predict(self, X):
        self.model.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            result = self.model(torch.tensor(X).to(self.device).float())
            return result.mean.data.cpu().numpy(), torch.sqrt(result.variance).data.cpu().numpy()

    def fit(self, epochs=50):
        self.model.train()
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        for i in range(epochs):
            self.optim.zero_grad()
            output = self.model(self.X)
            loss = -mll(output, self.Y)
            loss.backward()
            self.optim.step()
            

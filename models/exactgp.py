import gpytorch
import torch

class ExactGPModel(gpytorch.models.ExactGP):
    '''Exact GP model.'''

    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class FittedGP:
    '''GP on fixed X, Y data which can have hyperparameters tuned.'''

    def __init__(self, X, Y):
        self.X, self.Y = X, Y
        self.model = ExactGPModel(X, Y, gpytorch.likelihoods.GaussianLikelihood).cuda()
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.likelihood.train()
        self.optim = torch.optim.Adam(self.model.parameters(), lr=0.1)
        
    def predict(X):
        self.model.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            result = model(X)
            return result.mean.data.cpu().numpy(), torch.sqrt(result.variance).data.cpu().numpy()

    def fit(epochs=50):
        self.model.train()
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        for i in range(epochs):
            self.optim.zero_grad()
            output = self.model(self.X)
            loss = -mll(output, self.Y)
            loss.backward()
            self.optim.step()
            
        

        

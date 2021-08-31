import torch
import gpytorch

class GP(gpytorch.models.ExactGP):
    def __init__(self, in_dim, out_dim, likelihood):
        super().__init__(train_inputs=None,
                         train_targets=None,
                         likelihood=likelihood)
        self.out_dim = out_dim
        if self.out_dim == 1:
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        else:
            batch_shape = torch.Size([self.out_dim])
            self.mean_module = gpytorch.means.ConstantMean(batch_shape=batch_shape)
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(batch_shape=batch_shape),
                batch_shape=batch_shape
            )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        mvn = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        # ExactGP's forward method must return a distribution, you cannot return the sample from the distribution directly from this method
        return mvn if self.out_dim == 1 else gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(mvn)


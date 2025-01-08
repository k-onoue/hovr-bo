import gpytorch
from botorch.models import SingleTaskGP
from gpytorch.kernels import MaternKernel
from gpytorch.constraints import Interval
from gpytorch.priors import UniformPrior


class PredefinedSingleTaskGP(SingleTaskGP):
    def __init__(self, train_X, train_Y):
        # Initialize base class
        super().__init__(train_X, train_Y)

        # Replace the default kernel with the custom Matérn-5/2 kernel
        ard_num_dims = train_X.size(-1)  # Automatically determine ARD dims
        self.covar_module = gpytorch.kernels.ScaleKernel(
            base_kernel=MaternKernel(
                nu=2.5,
                ard_num_dims=ard_num_dims,
                lengthscale_prior=UniformPrior(0.005, 2.0),
                lengthscale_constraint=Interval(0.005, 2.0),
            ),
            outputscale_prior=UniformPrior(0.05, 20.0),
            outputscale_constraint=Interval(0.05, 20.0),
        )

        # Set the Gaussian likelihood with fixed noise settings
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_prior=UniformPrior(0.0005, 0.1),
            noise_constraint=Interval(0.0005, 0.1),
        )



# if __name__ == "__main__":
#     import torch
#     from botorch.fit import fit_gpytorch_mll
#     from gpytorch.mlls import ExactMarginalLogLikelihood

#     # Generate sample data
#     train_x = torch.linspace(0, 1, 10).view(-1, 1)  # 1D example
#     train_y = torch.sin(train_x * (2 * torch.pi)) + 0.2 * torch.randn_like(train_x)

#     # Initialize the model
#     model = PredefinedSingleTaskGP(train_x, train_y)

#     # Fit the model
#     mll = ExactMarginalLogLikelihood(model.likelihood, model)
#     fit_gpytorch_mll(mll)

#     # Test the model
#     model.eval()

#     test_x = torch.linspace(0, 1, 51).view(-1, 1)
#     with torch.no_grad(), gpytorch.settings.fast_pred_var():
#         preds = model.posterior(test_x).mean
#         lower, upper = model.posterior(test_x).confidence_region()

#     # Plot results
#     import matplotlib.pyplot as plt

#     plt.figure(figsize=(8, 6))
#     plt.plot(train_x.numpy(), train_y.numpy(), 'k*', label="Training Data")
#     plt.plot(test_x.numpy(), preds.numpy(), 'b', label="Predictive Mean")
#     plt.fill_between(
#         test_x.numpy().squeeze(),
#         lower.numpy(),
#         upper.numpy(),
#         alpha=0.5,
#         label="Confidence Region",
#     )
#     plt.legend()
#     plt.show()

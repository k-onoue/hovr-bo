from typing import Any, Callable, List, Optional

import numpy as np
import torch
from botorch.models.model import Model
from botorch.posteriors import Posterior
from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch import distributions as gdists
from laplace import Laplace
from torch import Tensor

from .utils import RegNet, get_best_hyperparameters
from .utils import augmented_and_regularized_trimmed_loss


class LaplacePosterior(Posterior):
    def __init__(self, posterior, output_dim):
        super().__init__()
        self.post = posterior
        self.output_dim = output_dim

    def rsample(
        self,
        sample_shape: Optional[torch.Size] = None,
    ) -> Tensor:
        samples = self.post.rsample(sample_shape).squeeze(-1)
        new_shape = samples.shape[:-1]
        return samples.reshape(*new_shape, -1, self.output_dim)

    @property
    def mean(self) -> Tensor:
        r"""The posterior mean."""
        post_mean = self.post.mean.squeeze(-1)
        shape = post_mean.shape
        return post_mean.reshape(*shape[:-1], -1, self.output_dim)

    @property
    def variance(self) -> Tensor:
        r"""The posterior variance."""
        post_var = self.post.variance.squeeze(-1)
        shape = post_var.shape
        return post_var.reshape(*shape[:-1], -1, self.output_dim)

    @property
    def device(self) -> torch.device:
        return self.post.device

    @property
    def dtype(self) -> torch.dtype:
        r"""The torch dtype of the distribution."""
        return self.post.dtype
    

class LaplaceBNN(Model):
    def __init__(self, args, input_dim, output_dim, device):
        super().__init__()

        self.regnet_dims = args["regnet_dims"]
        self.regnet_activation = args["regnet_activation"]
        self.prior_var = args["prior_var"] if "prior_var" in args else 1.0
        self.noise_var = args["noise_var"] if "noise_var" in args else torch.tensor(1.0)
        self.likelihood = "regression"
        self.iterative = args["iterative"] if "iterative" in args else False
        self.nn = RegNet(
            dimensions=self.regnet_dims,
            activation=self.regnet_activation,
            input_dim=input_dim,
            output_dim=output_dim,
            dtype=torch.float64,
            device=device
        )
        self.bnn = None
        self.output_dim = output_dim

    def posterior_predictive(self, X, bnn):
        if len(X.shape) < 3:
            B, D = X.shape
            Q = 1
        else:
            # Transform to `(batch_shape*q, d)`
            B, Q, D = X.shape
            X = X.reshape(B*Q, D)

        K = self.num_outputs
        # Posterior predictive distribution
        # mean_y is (batch_shape*q, k); cov_y is (batch_shape*q*k, batch_shape*q*k)
        mean_y, cov_y = self._get_prediction(X, bnn)

        # Mean in `(batch_shape, q*k)`
        mean_y = mean_y.reshape(B, Q*K)

        # Cov is `(batch_shape, q*k, q*k)`
        cov_y += 1e-4*torch.eye(B*Q*K).to(X)
        cov_y = cov_y.reshape(B, Q, K, B, Q, K)
        cov_y = torch.einsum('bqkbrl->bqkrl', cov_y)  # (B, Q, K, Q, K)
        cov_y = cov_y.reshape(B, Q*K, Q*K)

        dist = gdists.MultivariateNormal(mean_y, covariance_matrix=cov_y)
        post_pred = GPyTorchPosterior(dist)

        if K > 1 and Q > 1:
            return LaplacePosterior(post_pred, self.output_dim)
        else:        
            return post_pred

    def posterior(
        self,
        X: Tensor,
        output_indices: Optional[List[int]] = None,
        observation_noise: bool = False,
        posterior_transform: Optional[Callable[[Posterior], Posterior]] = None,
        **kwargs: Any,
    ) -> Posterior:
        return self.posterior_predictive(X, self.bnn)

    @property
    def num_outputs(self) -> int:
        return self.output_dim

    def _get_prediction(self, test_x: torch.Tensor, bnn):
        """
        Batched Laplace prediction.

        Args:
            test_x: Tensor of size `(batch_shape, d)`.

        Returns:
            Tensor of size `(batch_shape, k)`
        """
        mean_y, cov_y = bnn(test_x, joint=True)

        return mean_y, cov_y
    
    def get_likelihood(self, train_x, train_y, prior_var, noise_var):
        # fit to 80% of the data, and evaluate on the rest
        n = len(train_x)
        n_train = int(0.8 * n)
        train_x, val_x = train_x[:n_train], train_x[n_train:]
        train_y, val_y = train_y[:n_train], train_y[n_train:]
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(train_x, train_y),
        )

        model = self.fit_laplace(train_loader, prior_var, noise_var)
        posterior = self.posterior_predictive(val_x, model)
    
        predictions_mean = posterior.mean
        # std combines epistemic and aleatoric uncertainty
        predictions_std = torch.sqrt(posterior.variance + self.noise_var)
        # get log likelihood
        likelihood = torch.distributions.Normal(predictions_mean, predictions_std).log_prob(val_y).sum()
        return likelihood

    def fit_laplace(self, train_loader, prior_var, noise_var):
        bnn = Laplace(
            self.nn, 
            self.likelihood,
            sigma_noise=np.sqrt(noise_var),
            prior_precision=(1 / prior_var),
            subset_of_weights='all',
            hessian_structure='kron',
            enable_backprop=True
        )
        bnn.fit(train_loader)
        bnn.optimize_prior_precision(n_steps=50)

        return bnn

    def fit(self, train_x, original_train_y, loss_params=None):
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(train_x, original_train_y),
            batch_size=len(train_x), shuffle=True
        )

        # Use loss_params with default values
        if loss_params is None:
            loss_params = {}
        n_epochs = loss_params.get("n_epochs", 1000)
        weight_decay = loss_params.get("weight_decay", 0)
        artl_weight = loss_params.get("artl_weight", 1.0)  # Weight for ARTL loss
        h = loss_params.get("h", int(0.9 * len(train_x)))  # 90% of data points
        lambd = loss_params.get("lambd", 1e-3)  # Regularization strength
        k = loss_params.get("k", (1, 2, 3))  # Order of derivatives in HOVR
        q = loss_params.get("q", 2)  # Exponent in HOVR
        M = loss_params.get("M", 100)  # Number of random points for HOVR

        # optimizer = torch.optim.Adam(self.nn.parameters(), lr=1e-1, weight_decay=1e-3)
        optimizer = torch.optim.Adam(self.nn.parameters(), lr=1e-2, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs * len(train_loader))

        print(f"n_epochs: {n_epochs}")
        print(f"weight_decay: {weight_decay}")
        print(f"artl_weight: {artl_weight}")
        print(f"h: {h}")
        print(f"lambd: {lambd}")
        print(f"k: {k}")
        print(f"q: {q}")
        print(f"M: {M}")

        mse_loss_func = torch.nn.MSELoss()

        for epoch in range(n_epochs):
            for x, y in train_loader:
                optimizer.zero_grad()

                # Compute MSE loss
                mse_loss = mse_loss_func(self.nn(x), y)

                if artl_weight != 0:
                    # Compute augmented and regularized trimmed loss (ARTL)
                    artl_loss = augmented_and_regularized_trimmed_loss(
                        model=self.nn,
                        X_tensor=x,
                        y_tensor=y,
                        h=h,
                        lambd=lambd,
                        k=k,
                        q=q,
                        M=M
                    )
                else:
                    artl_loss = 0

                # Combine both losses with weights
                total_loss = mse_loss + artl_weight * artl_loss

                # Backpropagation and optimization
                total_loss.backward()
                optimizer.step()
                scheduler.step()

        self.nn.eval()

        if self.iterative:
            llh_fn = self.get_likelihood
            self.prior_var, self.noise_var = get_best_hyperparameters(train_x, original_train_y, llh_fn)

        self.bnn = self.fit_laplace(train_loader, self.prior_var, self.noise_var)

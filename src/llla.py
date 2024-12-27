import os
from typing import Any, Callable, List, Optional

import numpy as np
import torch
from botorch.models.model import Model
from botorch.posteriors import Posterior
from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch import distributions as gdists
from laplace import Laplace
from torch import Tensor

from .nn_utils import RegNet, get_best_hyperparameters
from .nn_utils import augmented_and_regularized_trimmed_loss
# from .nn_utils import EarlyStopping
from early_stopping_pytorch import EarlyStopping


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
    def __init__(
        self, 
        args: dict, 
        input_dim, 
        output_dim, 
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float64
    ):
        super().__init__()
        self.likelihood = "regression"
        self.regnet_dims = args.get("regnet_dims", [128, 128, 128])
        self.regnet_activation = args.get("regnet_activation", "tanh")
        self.prior_var = args.get("prior_var", 10.0)
        self.noise_var = args.get("noise_var", 1.0)
        self.iterative = args.get("iterative", True)
        self.loss_params = args.get("loss_params", {})
        self.nn = RegNet(
            dimensions=self.regnet_dims,
            activation=self.regnet_activation,
            input_dim=input_dim,
            output_dim=output_dim,
            dtype=dtype,
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
            X = X.reshape(B * Q, D)

        K = self.num_outputs
        # Posterior predictive distribution
        mean_y, cov_y = self._get_prediction(X, bnn)

        # Reshape mean
        mean_y = mean_y.reshape(B, Q * K)

        # Reshape covariance
        cov_y += 1e-4 * torch.eye(B * Q * K).to(X)
        cov_y = cov_y.reshape(B, Q, K, B, Q, K)
        cov_y = torch.einsum('bqkbrl->bqkrl', cov_y)  # (B, Q, K, Q, K)
        cov_y = cov_y.reshape(B, Q * K, Q * K)

        dist = gdists.MultivariateNormal(mean_y, covariance_matrix=cov_y)
        post_pred = GPyTorchPosterior(dist)

        # Return a custom LaplacePosterior if multiple outputs in a batched scenario
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
            Tuple of (mean, cov) with shapes:
            - mean: `(batch_shape, k)`
            - cov: `(batch_shape*k, batch_shape*k)`
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
            batch_size=min(32, len(train_x)),  # smaller batch size
            shuffle=True
        )

        model = self.fit_laplace(train_loader, prior_var, noise_var)
        posterior = self.posterior_predictive(val_x, model)
    
        predictions_mean = posterior.mean
        predictions_std = torch.sqrt(posterior.variance + self.noise_var)
        # get log likelihood
        likelihood = torch.distributions.Normal(predictions_mean, predictions_std).log_prob(val_y).sum()
        return likelihood

    def fit_laplace(self, train_loader, prior_var, noise_var):
        """
        Fit a Laplace approximation on the last layer of the neural network.
        """
        bnn = Laplace(
            self.nn, 
            self.likelihood,
            sigma_noise=np.sqrt(noise_var),
            prior_precision=(1 / prior_var),
            subset_of_weights='last_layer',
            hessian_structure='full',
            enable_backprop=True
        )
        bnn.fit(train_loader)
        bnn.optimize_prior_precision(n_steps=50)

        return bnn

    def fit(self, train_x, original_train_y, model_param_path=None):
        """
        Train the neural network (MSE + optional ARTL) and optionally fit Laplace approximation.
        """
        # Use a smaller batch_size to avoid training with the entire dataset at once
        batch_size = min(32, len(train_x))
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(train_x, original_train_y),
            batch_size=batch_size,
            shuffle=True
        )

        n_epochs = self.loss_params.get("n_epochs", 10000)
        lr = self.loss_params.get("lr", 1e-2)
        weight_decay = self.loss_params.get("weight_decay", 0)
        momentum = self.loss_params.get("momentum", 0)
        artl_weight = self.loss_params.get("artl_weight", 0)  # Weight for ARTL loss
        h = self.loss_params.get("h", int(0.9 * len(train_x)))
        lambd = self.loss_params.get("lambd", 1e-3)
        k = self.loss_params.get("k", (1, 2, 3))
        q = self.loss_params.get("q", 2)
        M = self.loss_params.get("M", 10)

        # Load model state if provided and the file exists
        if model_param_path and os.path.isfile(model_param_path):
            try:
                model_state = torch.load(model_param_path, weights_only=True)
                self.nn.load_state_dict(model_state)
            except EOFError:
                # File is empty or corrupted
                model_state = None

        # Use a simple SGD without scheduling
        optimizer = torch.optim.SGD(
            self.nn.parameters(), 
            lr=lr, 
            weight_decay=weight_decay, 
            momentum=momentum
        )
        mse_loss_func = torch.nn.MSELoss()

        early_stopping = EarlyStopping(patience=1000, verbose=True, path=model_param_path)

        for epoch in range(n_epochs):
            for x, y in train_loader:
                optimizer.zero_grad()

                # Compute MSE loss
                mse_loss = mse_loss_func(self.nn(x), y)

                # Compute ARTL loss if needed
                if artl_weight != 0:
                    artl_loss_val = augmented_and_regularized_trimmed_loss(
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
                    artl_loss_val = 0

                # Combine both losses
                total_loss = mse_loss + artl_weight * artl_loss_val

                # Backpropagation
                total_loss.backward()
                optimizer.step()

            # Logging
            mse_loss_value = mse_loss.item()
            artl_loss_value = (
                artl_loss_val.item() if isinstance(artl_loss_val, torch.Tensor) else artl_loss_val
            )

            print(f"Epoch {epoch+1}/{n_epochs}: MSE Loss: {mse_loss_value}, ARTL Loss: {artl_loss_value}")

            # Use training loss as "validation" loss for early stopping
            # val_loss = total_loss.item()
            val_loss = mse_loss_value
            early_stopping(val_loss, self.nn)

            if early_stopping.early_stop:
                print("Early stopping triggered")
                break

        # Reload the best model weights
        self.nn.load_state_dict(torch.load(model_param_path, weights_only=True))
        self.nn.eval()

        # Fit Laplace approximation if iterative
        if self.iterative:
            llh_fn = self.get_likelihood
            self.prior_var, self.noise_var = get_best_hyperparameters(
                train_x, original_train_y, llh_fn
            )

        self.bnn = self.fit_laplace(train_loader, self.prior_var, self.noise_var)

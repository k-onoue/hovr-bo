from typing import Union, Optional, Callable, Any

import torch
from botorch.models.model import Model
from botorch.posteriors import Posterior
from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch import distributions as gdists
from laplace import Laplace
from torch import Tensor
from torch.utils.data import DataLoader

from ._utils import EarlyStopping
from ._utils import MLP
from ._utils import hovr_loss_fn, trimmed_loss_fn


class LaplaceModel(Model):
    def __init__(
        self,
        dimensions: list[int],
        activation: str,
        input_dim: int,
        output_dim: int,
        dtype: torch.dtype = torch.float64,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        super().__init__()

        self.output_dim = output_dim

        self.nn = MLP(
            dimensions=dimensions,
            activation=activation,
            input_dim=input_dim,
            output_dim=output_dim,
            dtype=dtype,
            device=device,
        )

        self.bnn = None

    @property
    def num_outputs(self) -> int:
        return self.output_dim

    def forward(self, X: Tensor) -> Tensor:
        mean, covariance = self.bnn(X, joint=True)
        return mean, covariance
    
    def posterior(
        self,
        X: Tensor,
        output_indices: Optional[list[int]] = None,
        observation_noise: bool = False,
        posterior_transform: Optional[Callable[[Posterior], Posterior]] = None,
        **kwargs: Any,
    ) -> Posterior:
        """
        Compute the posterior predictive distribution.

        Args:
            X: Input tensor of shape (batch_shape, q, d) or (batch_shape*q, d).
            output_indices: List of output indices to consider (not used in this example).
            observation_noise: Flag for including observation noise (not implemented here).
            posterior_transform: Optional transformation to apply to the posterior.
            **kwargs: Additional arguments for customization.

        Returns:
            Posterior: GPyTorchPosterior object encapsulating the posterior predictive.
        """
        # Determine input shape and reshape if needed
        if len(X.shape) < 3:
            B, D = X.shape
            Q = 1
        else:
            B, Q, D = X.shape
            X = X.reshape(B * Q, D)

        # Generate predictions (mean and covariance)
        mean, covariance = self.forward(X)

        # Reshape mean to (B, Q * output_dim)
        mean = mean.reshape(B, Q * self.output_dim)

        # Add jitter to covariance matrix for numerical stability
        jitter = 1e-4 * torch.eye(covariance.shape[-1], device=covariance.device, dtype=covariance.dtype)
        covariance += jitter

        # Reshape covariance for batched processing
        K = self.output_dim
        covariance = covariance.reshape(B, Q, K, B, Q, K)
        covariance = torch.einsum('bqkbrl->bqkrl', covariance)  # Reshape to (B, Q, K, Q, K)
        covariance = covariance.reshape(B, Q * K, Q * K)

        # Create a multivariate normal distribution
        dist = gdists.MultivariateNormal(mean, covariance_matrix=covariance)
        posterior = GPyTorchPosterior(dist)

        return posterior

    def loss_fn(
        self,
        x: Tensor,
        y: Tensor,
        mse_coeff: float,
        trim_coeff: float,
        hovr_coeff: float,
        h: int = None,
        k: tuple[int, ...] = (1, 2),
        q: int = 2,
        M: int = 10,
    ) -> Tensor:
        """
        Computes the combined loss based on MSE, trimmed loss, and HOVR loss.

        Args:
            x (Tensor): Input features.
            y (Tensor): Target labels.
            mse_coeff (float): Coefficient for the MSE loss.
            trim_coeff (float): Coefficient for the trimmed loss.
            hovr_coeff (float): Coefficient for the HOVR loss.
            h (int, optional): Parameter for trimmed loss.
            k (tuple[int, ...], optional): Parameter for HOVR loss.
            q (int, optional): Parameter for HOVR loss.
            M (int, optional): Parameter for HOVR loss.

        Returns:
            Tensor: Combined loss value.
        """
        y_pred = self.nn(x)

        # Define individual loss components
        def calculate_loss(coeff: float, loss_fn: callable, *args, **kwargs) -> float:
            return coeff * loss_fn(*args, **kwargs) if coeff > 0 else 0

        # Loss calculations
        mse_loss = calculate_loss(mse_coeff, torch.nn.MSELoss(), y_pred, y)
        trim_loss = calculate_loss(trim_coeff, trimmed_loss_fn, self.nn, x, y, h=h)
        hovr_loss = calculate_loss(hovr_coeff, hovr_loss_fn, self.nn, x, y, k=k, q=q, M=M)

        # Combine losses
        total_loss = mse_loss + trim_loss + hovr_loss
        
        return total_loss

    def fit(
        self,
        x: Tensor,
        y: Tensor,
        config: dict = {}
    ) -> None:
        """
        MAP Estimation (deterministic training) followed by Uncertainty Estimation via Laplace approximation.

        config = {
            # Data loading
            "batch_size": 32,          # Batch size for training
            "val_split": 0.2,          # Validation split ratio
            "min_train_size": 5,       # Minimum dataset size for validation split

            # Optimization
            "optimizer": torch.optim.Adam,  # Optimizer class
            "lr": 1e-3,                    # Learning rate
            "weight_decay": 0,             # Weight decay for non-VBLL layers
            "epochs": 1000,                 # Number of training epochs

            # Loss coefficients
            "loss_coeffs": {
                "mse": 1,                 # MSE loss coefficient
                "trim": 0,                # Trimmed loss coefficient
                "hovr": 0,                # HOVR loss coefficient
            }

            # Loss parameters
            loss_params = {
                "h": None,                # Number of points to keep after trimming
                "k": (1, 2),              # Tuple of derivative orders
                "q": 2,                   # Exponent in HOVR
                "M": 10,                  # Number of random points
            }

            # Early Stopping
            "patience": 20,           # Number of epochs to wait for improvement
            "verbose": True,         # Print early stopping messages
            "delta": 0,              # Minimum change to qualify as improvement

            # Laplace approximation
            "hessian_structure": "full",
            "prior_precision": 1e-2,  # Too small prior precision can lead to numerical instability
            "sigma_noise": 1e-1,
            "temperature": 1,
        }
        """
        train_loader = self.fit_map(x, y, config) # Share train loader with Laplace
        
        self.bnn = Laplace(
            model=self.nn,
            likelihood="regression",
            subset_of_weights='last_layer',
            hessian_structure=config.get("hessian_structure", "full"),
            prior_precision=config.get("prior_precision", 1e-2),
            sigma_noise=config.get("sigma_noise", 1e-1),
            temperature=config.get("temperature", 1),
            enable_backprop=True
        )

        self.bnn.fit(train_loader)

    def fit_map(
        self,
        x: Tensor,
        y: Tensor,
        config: dict
    ) -> None:
        """
        Reference fit method.
        """

        # Retrieve minimum training size for Early Stopping
        min_train_size = config.get("min_train_size", 5)
        
        dataset_size = x.size(0)
        
        if dataset_size >= min_train_size:
            # Split data into training and validation sets
            val_split = config.get("val_split", 0.2)
            indices = torch.randperm(dataset_size)
            split = int(val_split * dataset_size)
            train_indices, val_indices = indices[split:], indices[:split]
            train_dataset = torch.utils.data.TensorDataset(x[train_indices], y[train_indices])
            val_dataset = torch.utils.data.TensorDataset(x[val_indices], y[val_indices])
            train_loader = DataLoader(train_dataset, batch_size=config.get("batch_size", 32), shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=config.get("batch_size", 32), shuffle=False)

            # Initialize EarlyStopping
            early_stopping = EarlyStopping(
                patience=config.get("patience", 20),
                verbose=config.get("verbose", True),
                delta=config.get("delta", 0)
            )

        else:
            # Use the entire dataset for training without validation
            train_dataset = torch.utils.data.TensorDataset(x, y)
            train_loader = DataLoader(train_dataset, batch_size=config.get("batch_size", 32), shuffle=True)
            val_loader = None  # No validation
            early_stopping = None  # EarlyStopping not applied

        weight_decay = config.get("weight_decay", 0)

        loss_coeffs = config.get("loss_coeffs", {})
        mse_coeff = loss_coeffs.get("mse", 1)
        trim_coeff = loss_coeffs.get("trim", 0)
        hovr_coeff = loss_coeffs.get("hovr", 0)
        total_coeff = mse_coeff + trim_coeff + hovr_coeff + weight_decay
        mse_coeff /= total_coeff
        trim_coeff /= total_coeff
        hovr_coeff /= total_coeff
        weight_decay /= total_coeff

        loss_params = config.get("loss_params", {})
        h = loss_params.get("h", None)
        k = loss_params.get("k", (1, 2))
        q = loss_params.get("q", 2)
        M = loss_params.get("M", 10)

        non_out_layer_params = []
        out_layer_params = []
        for name, param in self.named_parameters():
            if name.startswith("nn.out_layer"):
                out_layer_params.append(param)
            else:
                non_out_layer_params.append(param)

        param_list = [
            {"params": non_out_layer_params, "weight_decay": weight_decay},
            {"params": out_layer_params, "weight_decay": 0},
        ]

        optimizer_class = config.get("optimizer", torch.optim.Adam)
        optimizer = optimizer_class(param_list, lr=config.get("lr", 1e-3))

        epochs = config.get("epochs", 1000)
        for _ in range(epochs):
            self.train()
            for x_batch, y_batch in train_loader:

                optimizer.zero_grad()
                batch_loss = self.loss_fn(
                    x_batch, 
                    y_batch, 
                    mse_coeff, 
                    trim_coeff, 
                    hovr_coeff,
                    h=h,
                    k=k,
                    q=q,
                    M=M
                )
                batch_loss.backward()
                optimizer.step()

            if early_stopping and val_loader is not None:
                # Validation
                self.eval()
                val_loss = 0
                with torch.no_grad():
                    for x_val, y_val in val_loader:
                        loss = self.loss_fn(
                            x_val, 
                            y_val,
                            mse_coeff,
                            trim_coeff,
                            hovr_coeff,
                            h=h,
                            k=k,
                            q=q,
                            M=M
                        )
                        val_loss += loss.item()
                val_loss /= len(val_loader)

                # Check early stopping
                early_stopping(val_loss, self)
                if early_stopping.early_stop:
                    break
        
        return train_loader
    


# # Example testing the LaplaceModel with 1D regression
# if __name__ == "__main__":
#     import numpy as np
#     import plotly.graph_objects as go

#     from _utils import EarlyStopping
#     from _utils import MLP
#     from _utils import hovr_loss_fn, trimmed_loss_fn


#     # Generate synthetic data
#     np.random.seed(0)
#     torch.manual_seed(0)

#     X = np.linspace(-5, 5, 30).reshape(-1, 1)
#     y = np.sin(X) + 0.2 * np.random.normal(size=X.shape)

#     X_train = torch.tensor(X, dtype=torch.float64)
#     y_train = torch.tensor(y, dtype=torch.float64)

#     # Define model configuration
#     model = LaplaceModel(
#         dimensions=[128, 128, 128],
#         activation="tanh",
#         input_dim=1,
#         output_dim=1,
#     )

#     config = {
#         "batch_size": 16,
#         "epochs": 5000,
#         "prior_precision": 1, # too small prior precision can lead to numerical instability
#         "sigma_noise": 1e-1,
#         "loss_coeffs": {"mse": 1, "trim": 1e-1, "hovr": 1e-3},
#     }

#     # Train model
#     model.fit(X_train, y_train, config)

#     # Predict
#     X_test = torch.linspace(-6, 6, 200).reshape(-1, 1).to(torch.float64)
#     # temp = model.posterior(X_test)
#     # print(temp)


#     with torch.no_grad():
#         y_pred, covariance = model.forward(X_test)
#         jitter = torch.eye(covariance.shape[0]) * 1e-4
#         covariance += jitter
#         temp = torch.linalg.cholesky(covariance)

#         print(temp)

#         print(covariance)

#     # Convert to numpy for plotting
#     X_test_np = X_test.numpy()
#     y_pred_np = y_pred.numpy().squeeze()
#     std_dev = torch.sqrt(torch.diagonal(covariance, dim1=-2, dim2=-1)).numpy().squeeze()

#     # Plot results
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(x=X.squeeze(), y=y.squeeze(), mode="markers", name="Training Data"))
#     fig.add_trace(go.Scatter(x=X_test_np.squeeze(), y=y_pred_np, mode="lines", name="Prediction"))

#     # Add 2-sigma confidence intervals
#     fig.add_trace(go.Scatter(
#         x=X_test_np.squeeze(),
#         y=(y_pred_np + 2 * std_dev),
#         mode="lines",
#         name="Upper 2-sigma",
#         line=dict(dash="dash")
#     ))
#     fig.add_trace(go.Scatter(
#         x=X_test_np.squeeze(),
#         y=(y_pred_np - 2 * std_dev),
#         mode="lines",
#         name="Lower 2-sigma",
#         line=dict(dash="dash")
#     ))

#     fig.update_layout(
#         title="1D Function Regression with Laplace Model",
#         xaxis_title="Input",
#         yaxis_title="Output",
#     )
#     fig.show()
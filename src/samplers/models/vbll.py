from typing import Union, Optional, Callable, Any

import torch
import vbll
from botorch.models.model import Model
from botorch.posteriors import Posterior
from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch import distributions as gdists
from torch import nn
from torch import Tensor
from torch.utils.data import DataLoader

from ._utils import EarlyStopping


class VBLLMLP(nn.Sequential):
    def __init__(
        self,
        dimensions: list[int],
        activation: str,
        input_dim: int = 1,
        output_dim: int = 1,
        dtype: torch.dtype = torch.float64,
        device: Union[str, torch.device] = "cpu",
        reg_weight: float = 1.0,
        parameterization: str = "dense",
        prior_scale: float = 1.0,
        wishart_scale: float = 0.1,
    ) -> None:
        super(VBLLMLP, self).__init__()
        
        # Store dimensions for all layers
        self.dimensions = [input_dim, *dimensions]  # Exclude output_dim

        # Create hidden layers
        for i in range(len(self.dimensions) - 1):
            # Add linear layer
            self.add_module(
                f"linear{i}",
                nn.Linear(
                    self.dimensions[i],
                    self.dimensions[i + 1],
                    dtype=dtype,
                    device=device,
                ),
            )

            # Add activation function
            if activation == "tanh":
                self.add_module(f"activation{i}", nn.Tanh())
            elif activation == "relu":
                self.add_module(f"activation{i}", nn.ReLU())
            elif activation == "elu":
                self.add_module(f"activation{i}", nn.ELU())
            else:
                raise NotImplementedError(
                    f"Activation type '{activation}' is not supported."
                )

        # Add VBLL as final layer
        self.out_layer = vbll.Regression(
            self.dimensions[-1],  # Use last hidden dim
            output_dim,
            reg_weight,
            parameterization=parameterization,
            prior_scale=prior_scale,
            wishart_scale=wishart_scale,
        ).to(dtype=dtype, device=device)

    def forward(self, x: torch.Tensor) -> vbll.layers.regression.VBLLReturn:
        for name, module in self.named_children():
            if name != "out_layer":
                x = module(x)
        return self.out_layer(x)


class VBLLModel(Model):
    def __init__(
        self,
        dimensions: list[int],
        activation: str,
        input_dim: int,
        output_dim: int,
        reg_weight: float,
        parameterization: str,
        prior_scale: float,
        wishart_scale: float,
        dtype: torch.dtype = torch.float64,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        """
        A VBLL model for regression.

        Args:
            dimensions (list[int]): List of hidden layer dimensions for the regression network.
            activation (str): Activation function for the regression network ('tanh', 'relu', 'elu').
            input_dim (int): Dimension of the input.
            output_dim (int): Dimension of the output.
            reg_weight (float): Regularization weight for VBLL.
            parameterization (str): parameterization for VBLL. {'dense', 'diagonal', 'lowrank', 'dense_precision'}.
            prior_scale (float): Prior scale for VBLL.
            wishart_scale (float): Wishart scale for VBLL.
            dtype (torch.dtype): Data type for the model's parameters.
            device (Union[str, torch.device]): Device for the model.
        """
        super().__init__()

        self.output_dim = output_dim

        # Define the regression network
        self.nn = VBLLMLP(
            dimensions,
            activation,
            input_dim,
            output_dim,
            dtype=dtype,
            device=device,
            reg_weight=reg_weight,
            parameterization=parameterization,
            prior_scale=prior_scale,
            wishart_scale=wishart_scale,
        )

    def forward(self, x: torch.Tensor) -> torch.distributions.Normal:
        """
        Perform a forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            vbll.VBLLReturn: Predictive distribution and loss functions from VBLL.
        """
        return self.nn(x)
    
    def posterior(
        self,
        X: Tensor,
        output_indices: Optional[list[int]] = None,
        observation_noise: bool = False,
        posterior_transform: Optional[Callable[[Posterior], Posterior]] = None,
        **kwargs: Any,
    ) -> Posterior:
        posterior = self.forward(X).predictive
        mean = posterior.mean.squeeze(-1)
        variance = posterior.variance.squeeze(-1)
        covariance = torch.diag_embed(variance)
        dist = gdists.MultivariateNormal(mean, covariance)
        posterior = GPyTorchPosterior(dist)
        return posterior

    @property
    def num_outputs(self) -> int:
        return self.output_dim

    def loss_fn(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.forward(x).train_loss_fn(y)
    
    def fit(
        self, 
        x: torch.Tensor, 
        y: torch.Tensor, 
        config: dict = {}
    ) -> None:
        """
        config = {
            # Data loading
            "batch_size": 32,          # Batch size for training
            "val_split": 0.2,          # Validation split ratio
            "min_train_size": 5,       # Minimum dataset size for validation split

            # Optimization
            "optimizer": torch.optim.Adam,  # Optimizer class
            "lr": 1e-3,                    # Learning rate
            "weight_decay": 0,             # Weight decay for non-VBLL layers
            "epochs": 100,                 # Number of training epochs

            # Early Stopping
            "patience": 20,           # Number of epochs to wait for improvement
            "verbose": True,         # Print early stopping messages
            "delta": 0,              # Minimum change to qualify as improvement
        }
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

        non_out_layer_params = []
        out_layer_params = []
        for name, param in self.named_parameters():
            if name.startswith("nn.out_layer"):
                out_layer_params.append(param)
            else:
                non_out_layer_params.append(param)

        param_list = [
            {"params": non_out_layer_params, "weight_decay": config.get("weight_decay", 0)},
            {"params": out_layer_params, "weight_decay": 0},
        ]

        optimizer_class = config.get("optimizer", torch.optim.Adam)
        optimizer = optimizer_class(param_list, lr=config.get("lr", 1e-3))

        epochs = config.get("epochs", 1000)
        for _ in range(epochs):
            self.train()
            for x_batch, y_batch in train_loader:

                optimizer.zero_grad()
                batch_loss = self.loss_fn(x_batch, y_batch)
                batch_loss.backward()
                optimizer.step()

            if early_stopping and val_loader is not None:
                # Validation
                self.eval()
                val_loss = 0
                with torch.no_grad():
                    for x_val, y_val in val_loader:
                        loss = self.loss_fn(x_val, y_val)
                        val_loss += loss.item()
                val_loss /= len(val_loader)

                # Check early stopping
                early_stopping(val_loss, self)
                if early_stopping.early_stop:
                    break

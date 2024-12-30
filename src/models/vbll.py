from typing import Union

import torch
import models.vbll as vbll
from botorch.models.model import Model
from botorch.posteriors import Posterior
from botorch.posteriors.gpytorch import GPyTorchPosterior
from early_stopping_pytorch import EarlyStopping
from torch.utils.data import DataLoader

from .mlp import MLP


class VBLLMLP(MLP):
    def __init__(
        self,
        dimensions: list[int],
        activation: str,
        input_dim: int = 1,
        output_dim: int = 1,
        dtype: torch.dtype = torch.float64,
        device: Union[str, torch.device] = "cpu",
        reg_weight: float = 1.0,
        parametrization: str = "dense",
        prior_scale: float = 1.0,
        wishart_scale: float = 0.1,
    ) -> None:
        """
        A Multi-Layer Perceptron (MLP) with a VBLL last layer.

        Args:
            dimensions (list[int]): List of hidden layer dimensions.
            activation (str): Activation function ('tanh', 'relu', 'elu').
            input_dim (int): Dimension of the input.
            output_dim (int): Dimension of the output.
            dtype (torch.dtype): Data type for the model's parameters.
            device (Union[str, torch.device]): Device for the model.
            reg_weight (float): Regularization weight for VBLL.
            prior_scale (float): Prior scale for VBLL.
            wishart_scale (float): Wishart scale for VBLL.
        """
        super().__init__(dimensions, activation, input_dim, output_dim, dtype, device)

        # Replace the last layer with VBLL regression
        last_layer_input_dim = self.dimensions[-2]
        self.out_layer = vbll.Regression(
            last_layer_input_dim,
            output_dim,
            reg_weight,
            parametrization=parametrization,
            prior_scale=prior_scale,
            wishart_scale=wishart_scale,
        )

    def forward(self, x: torch.Tensor) -> vbll.VBLLReturn:
        """
        Perform a forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            vbll.VBLLReturn: Predictive distribution and loss functions from VBLL.
        """
        for name, module in self.named_children():
            # Pass through all layers except the last VBLL layer
            if name != "out_layer":
                x = module(x)

        # Pass through the VBLL regression layer
        return self.out_layer(x)


class VBLLModel(Model):
    def __init__(
        self,
        dimensions: list[int],
        activation: str,
        input_dim: int,
        output_dim: int,
        reg_weight: float,
        parammetrization: str,
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
            parammetrization (str): Parametrization for VBLL. {'dense', 'diagonal', 'lowrank', 'dense_precision'}.
            prior_scale (float): Prior scale for VBLL.
            wishart_scale (float): Wishart scale for VBLL.
            dtype (torch.dtype): Data type for the model's parameters.
            device (Union[str, torch.device]): Device for the model.
        """
        super().__init__()

        # Define the regression network
        self.nn = VBLLMLP(
            dimensions,
            activation,
            input_dim,
            output_dim,
            dtype=dtype,
            device=device,
            reg_weight=reg_weight,
            parametrization=parammetrization,
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
        return self.nn(x).predictive
    
    def loss_fn(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.nn.forward(x).loss_fn(y)

    def fit(self, train_loader: DataLoader, config: dict = {}) -> None:
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

        epochs = config.get("epochs", 1)
        for _ in range(epochs):
            for x_batch, y_batch in train_loader:
                optimizer.zero_grad()
                batch_loss = self.loss(x_batch, y_batch)
                batch_loss.backward()
                optimizer.step()

    def posterior(self, x: torch.Tensor) -> Posterior:
        pred_dist = self.forward(x)
        return GPyTorchPosterior(pred_dist)

    @property
    def num_outputs(self) -> int:
        return self.output_dim
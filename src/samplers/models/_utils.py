from typing import Union

import torch
from torch import nn
import numpy as np


class MLP(nn.Sequential):
    def __init__(
        self,
        dimensions: list[int],
        activation: str,
        input_dim: int = 1,
        output_dim: int = 1,
        dtype: torch.dtype = torch.float64,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        """
        A Multi-Layer Perceptron (MLP) model.

        Args:
            dimensions (list[int]): List of hidden layer dimensions.
            activation (str): Activation function ('tanh', 'relu', 'elu').
            input_dim (int): Dimension of the input.
            output_dim (int): Dimension of the output.
            dtype (torch.dtype): Data type for the model's parameters.
            device (Union[str, torch.device]): Device for the model.
        """
        super(MLP, self).__init__()

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
        
        # Add output layer
        self.add_module(
            "out_layer", 
            nn.Linear(self.dimensions[-1], output_dim, dtype=dtype, device=device)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        """
        for _, module in self.named_children():
            x = module(x)
        return x


def trimmed_loss_fn(model, X_tensor, y_tensor, h=None):
    """
    Compute the Trimmed Loss.
    
    Parameters:
    - model: Neural network model
    - X_tensor: Input data (torch.Tensor)
    - y_tensor: Target data (torch.Tensor) 
    - h: Number of points to keep after trimming (default: 90% of data)
    
    Returns:
    - trim_loss: Computed trimmed loss
    """
    if h is None or h >= X_tensor.shape[0]:
        h = int(0.9 * X_tensor.shape[0])

    xi = nn.Parameter(torch.zeros(X_tensor.shape[0], 1), requires_grad=True)
    preds = model(X_tensor)
    residuals = (y_tensor - preds).view(-1, 1)
    loss_fit = (1 / X_tensor.shape[0]) * torch.sum((residuals - xi) ** 2)
    xi_squared = xi.view(-1) ** 2
    T_h_xi = (1 / X_tensor.shape[0]) * torch.sum(torch.topk(xi_squared, h, largest=False)[0])
    
    return loss_fit + T_h_xi


def hovr_loss_fn(model, X_tensor, _, k=(1, 2), q=2, M=10):
    """Compute the HOVR Loss."""
    # Skip HOVR computation during validation
    if not model.training:
        return torch.tensor(0.0, device=X_tensor.device)
    
    x_min, x_max = X_tensor.min(0)[0], X_tensor.max(0)[0]
    
    # Generate random points
    random_points = torch.rand(
        (M, X_tensor.shape[1]), 
        dtype=torch.double, 
        device=X_tensor.device, 
        requires_grad=True
    )
    random_points = random_points * (x_max - x_min) + x_min

    with torch.set_grad_enabled(True):  # Ensure gradients are enabled
        preds_random = model(random_points)
        
        # First-order gradients
        grads = torch.autograd.grad(
            preds_random, 
            random_points,
            torch.ones_like(preds_random), 
            create_graph=True
        )[0]

        hovr_term = 0.0
        n_dims = X_tensor.shape[1]

        # Higher-order gradients
        for order in k:
            temp_grads = grads
            for _ in range(order - 1):
                temp_grads = torch.autograd.grad(
                    temp_grads,
                    random_points, 
                    torch.ones_like(temp_grads),
                    create_graph=True
                )[0]
            hovr_term += (1 / n_dims) * torch.sum(torch.abs(temp_grads) ** q)

    return hovr_term


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_val_loss = None
        self.early_stop = False
        self.delta = delta
        self.trace_func = trace_func
        self.best_model_state = None  # Hold the best model state in memory

    def __call__(self, val_loss, model):
        # Check if validation loss is nan
        if np.isnan(val_loss):
            self.trace_func("Validation loss is NaN. Ignoring this epoch.")
            return

        if self.best_val_loss is None:
            self.best_val_loss = val_loss
            self.save_checkpoint(val_loss, model)
        elif val_loss < self.best_val_loss - self.delta:
            # Significant improvement detected
            self.best_val_loss = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0  # Reset counter since improvement occurred
        else:
            # No significant improvement
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, val_loss, model):
        """Saves model state in memory when validation loss decreases."""
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.best_val_loss:.6f} --> {val_loss:.6f}). Saving model state...')
        self.best_model_state = model.state_dict()  # Save the model state in memory


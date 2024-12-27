"""Likelihoods, priors, and RegNet."""
import torch
from torch import nn
import numpy as np


class RegNet(torch.nn.Sequential):
    def __init__(self, dimensions, activation, input_dim=1, output_dim=1,
                        dtype=torch.float64, device="cpu"):
        super(RegNet, self).__init__()
        self.dimensions = [input_dim, *dimensions, output_dim]
        for i in range(len(self.dimensions) - 1):
            self.add_module('linear%d' % i, torch.nn.Linear(
                self.dimensions[i], self.dimensions[i + 1], dtype=dtype, device=device)
            )
            if i < len(self.dimensions) - 2:
                if activation == "tanh":
                    self.add_module('tanh%d' % i, torch.nn.Tanh())
                elif activation == "relu":
                    self.add_module('relu%d' % i, torch.nn.ReLU())
                else:
                    raise NotImplementedError("Activation type %s is not supported" % activation)
                

def get_best_hyperparameters(train_x, train_y, likelihood_fn):
    # Define a range of values to test for prior_var and noise_var
    prior_var_values = torch.logspace(-2, 1, steps=5)  # Prior variance values from 0.01 to 10
    noise_var_values = torch.logspace(-2, 1, steps=5)  # Noise variance values from 0.01 to 10

    best_prior_var = None
    best_noise_var = None
    best_likelihood = -float('inf')

    # Perform grid search over prior_var and noise_var values
    for prior_var in prior_var_values:
        for noise_var in noise_var_values:
            # Calculate the log likelihood for the given hyperparameters
            likelihood = likelihood_fn(train_x, train_y, prior_var, noise_var)
            
            # Check if this is the best likelihood encountered so far
            if likelihood > best_likelihood:
                best_likelihood = likelihood
                best_prior_var = prior_var
                best_noise_var = noise_var

    return best_prior_var, best_noise_var


def augmented_and_regularized_trimmed_loss(
    model, X_tensor, y_tensor, h=None, lambd=1e-3, k=(1, 2), q=2, M=10
):
    """
    Compute the Augmented and Regularized Trimmed Loss (ARTL) with HOVR regularization.
    Automatically generates and uses xi (trimmed residual parameter).

    Parameters:
    - model: Neural network model.
    - X_tensor: Input data (torch.Tensor).
    - y_tensor: Target data (torch.Tensor).
    - h (int, optional): Number of points to keep after trimming. Defaults to 90% of data points.
    - lambd (float, optional): Regularization strength for HOVR term (default: 1e-3).
    - k (tuple, optional): Tuple or list of derivative orders to include in HOVR (default: (1, 2)).
    - q (int, optional): Exponent in HOVR (default: 2).
    - M (int, optional): Number of random points for HOVR evaluation (default: 10).

    Returns:
    - total_loss: Computed augmented trimmed loss.
    """
    # Default value for h if not provided
    if h is None:
        h = int(0.9 * X_tensor.shape[0])  # 90% of the data points

    # TTL computation
    if lambd < 1:  # Compute TTL only if tradeoff is not 1
        xi = nn.Parameter(torch.zeros(X_tensor.shape[0], 1), requires_grad=True)
        preds = model(X_tensor)
        residuals = (y_tensor - preds).view(-1, 1)
        loss_fit = (1 / X_tensor.shape[0]) * torch.sum((residuals - xi) ** 2)
        xi_squared = xi.view(-1) ** 2
        T_h_xi = (1 / X_tensor.shape[0]) * torch.sum(torch.topk(xi_squared, h, largest=False)[0])
        ttl_loss = loss_fit + T_h_xi
    else:
        ttl_loss = 0  # Skip TTL computation if tradeoff is 1

    # HOVR computation
    if lambd > 0:  # Compute HOVR only if tradeoff is not 0
        x_min, x_max = X_tensor.min(0)[0], X_tensor.max(0)[0]
        random_points = torch.tensor(
            np.random.uniform(x_min.numpy(), x_max.numpy(), (M, X_tensor.shape[1])),
            dtype=torch.double, requires_grad=True
        )
        preds_random = model(random_points)
        grads = torch.autograd.grad(preds_random, random_points, torch.ones_like(preds_random), create_graph=True)[0]

        hovr_term = 0.0
        n_dims = X_tensor.shape[1]

        for order in k:  # Iterate over specified derivative orders
            temp_grads = grads
            for _ in range(order - 1):  # Compute higher-order derivatives
                temp_grads = torch.autograd.grad(temp_grads, random_points, torch.ones_like(temp_grads),
                                                 create_graph=True)[0]
            # Accumulate HOVR terms
            hovr_term += (1 / n_dims) * torch.sum(torch.abs(temp_grads) ** q)

        hovr_loss = hovr_term
    else:
        hovr_loss = 0  # Skip HOVR computation if tradeoff is 0

    # Combine TTL and HOVR with tradeoff
    total_loss = ttl_loss + lambd * hovr_loss
    return total_loss


# class EarlyStopping:
#     """Early stops the training if validation loss doesn't improve after a given patience."""
#     def __init__(self, patience=7, verbose=False, delta=0, trace_func=print):
#         """
#         Args:
#             patience (int): How long to wait after last time validation loss improved.
#                             Default: 7
#             verbose (bool): If True, prints a message for each validation loss improvement.
#                             Default: False
#             delta (float): Minimum change in the monitored quantity to qualify as an improvement.
#                             Default: 0
#             trace_func (function): trace print function.
#                             Default: print
#         """
#         self.patience = patience
#         self.verbose = verbose
#         self.counter = 0
#         self.best_val_loss = None
#         self.early_stop = False
#         self.delta = delta
#         self.trace_func = trace_func
#         self.best_model_state = None  # Hold the best model state in memory

#     def __call__(self, val_loss, model):
#         # Check if validation loss is nan
#         if np.isnan(val_loss):
#             self.trace_func("Validation loss is NaN. Ignoring this epoch.")
#             return

#         if self.best_val_loss is None:
#             self.best_val_loss = val_loss
#             self.save_checkpoint(val_loss, model)
#         elif val_loss < self.best_val_loss - self.delta:
#             # Significant improvement detected
#             self.best_val_loss = val_loss
#             self.save_checkpoint(val_loss, model)
#             self.counter = 0  # Reset counter since improvement occurred
#         else:
#             # No significant improvement
#             self.counter += 1
#             self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
#             if self.counter >= self.patience:
#                 self.early_stop = True

#     def save_checkpoint(self, val_loss, model):
#         """Saves model state in memory when validation loss decreases."""
#         if self.verbose:
#             self.trace_func(f'Validation loss decreased ({self.best_val_loss:.6f} --> {val_loss:.6f}). Saving model state...')
#         self.best_model_state = model.state_dict()  # Save the model state in memory
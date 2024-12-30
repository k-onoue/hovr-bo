import os
from typing import Callable

import torch
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.sampling import SobolQMCNormalSampler


# -- LLLA -------------------------------------------
import plotly.graph_objects as go
from .models.llla import LaplaceBNN

def llla_artl_sampler(
    train_X: torch.Tensor,
    train_Y: torch.Tensor,
    bounds: torch.Tensor,
    batch_size: int = 1,
    mc_acqf: Callable = qLogExpectedImprovement,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float64,
    model_param_path: str = None,
    plot: bool = True,
    **kwargs
) -> torch.Tensor:
    r"""
    Sample candidates using the Laplace acquisition function.

    Args:
        train_X: A `n x d` tensor of training points.
        train_Y: A `n x 1` tensor of training targets.
        bounds: A `2 x d` tensor of lower and upper bounds for each dimension.
        batch_size: Number of candidates to sample.
        plot: Whether to visualize model fit and credible intervals (default: True).

    Returns:
        A `batch_size x d` tensor of candidate points.
    """
    input_dim = train_X.shape[-1]
    output_dim = train_Y.shape[-1]

    surrogate_model = LaplaceBNN(
        kwargs,
        input_dim=input_dim,
        output_dim=output_dim,
        device=device,
        dtype=dtype,
    )
    surrogate_model.fit(train_X, train_Y, model_param_path=model_param_path)

    if model_param_path:
        # Save the model state
        torch.save(surrogate_model.nn.state_dict(), model_param_path)

    acquisition_function = mc_acqf(
        model=surrogate_model,
        best_f=train_Y.max(),
        sampler=SobolQMCNormalSampler(sample_shape=torch.Size([500])),
    )

    candidates, _ = optimize_acqf(
        acq_function=acquisition_function,
        bounds=bounds,
        q=batch_size,
        num_restarts=10,
        raw_samples=50,
    )

    # Visualization
    if plot and input_dim == 1:
        test_X = torch.linspace(bounds[0, 0], bounds[1, 0], steps=500).to(device, dtype)
        posterior = surrogate_model.posterior(test_X.unsqueeze(-1))
        pred_mean = posterior.mean.detach().cpu().numpy().flatten()
        pred_std = torch.sqrt(posterior.variance).detach().cpu().numpy().flatten()

        # Compute credible intervals
        lower_bound = pred_mean - 2 * pred_std
        upper_bound = pred_mean + 2 * pred_std

        # Convert data for plotting
        train_X_np = train_X.cpu().numpy().flatten()
        train_Y_np = train_Y.cpu().numpy().flatten()
        test_X_np = test_X.cpu().numpy().flatten()

        # Plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=train_X_np, y=train_Y_np, mode='markers', name='Training Data'))
        fig.add_trace(go.Scatter(x=test_X_np, y=pred_mean, mode='lines', name='Prediction'))
        fig.add_trace(go.Scatter(x=test_X_np, y=upper_bound, mode='lines', line=dict(dash='dot'), name='Upper CI'))
        fig.add_trace(go.Scatter(x=test_X_np, y=lower_bound, mode='lines', line=dict(dash='dot'), name='Lower CI'))
        fig.update_layout(
            title="Laplace BNN Fit with Credible Intervals",
            xaxis_title="Input X",
            yaxis_title="Output Y",
        )
        fig.show()

    return candidates


def llla_l2_sampler(
    train_X: torch.Tensor,
    train_Y: torch.Tensor,
    bounds: torch.Tensor,
    batch_size: int = 1,
    mc_acqf: Callable = qLogExpectedImprovement,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float64,
    model_param_path: str = None,
    plot: bool = False,
    **kwargs
) -> torch.Tensor:
    r"""
    Sample candidates using the Laplace acquisition function.

    Args:
        train_X: A `n x d` tensor of training points.
        train_Y: A `n x 1` tensor of training targets.
        bounds: A `2 x d` tensor of lower and upper bounds for each dimension.
        batch_size: Number of candidates to sample.
        plot: Whether to visualize model fit and credible intervals (default: True).

    Returns:
        A `batch_size x d` tensor of candidate points.
    """
    input_dim = train_X.shape[-1]
    output_dim = train_Y.shape[-1]

    surrogate_model = LaplaceBNN(
        kwargs,
        input_dim=input_dim,
        output_dim=output_dim,
        device=device,
        dtype=dtype,
    )
    surrogate_model.fit(train_X, train_Y, model_param_path=model_param_path)

    if model_param_path:
        # Save the model state
        torch.save(surrogate_model.nn.state_dict(), model_param_path)

    acquisition_function = mc_acqf(
        model=surrogate_model,
        best_f=train_Y.max(),
        sampler=SobolQMCNormalSampler(sample_shape=torch.Size([500])),
    )

    candidates, _ = optimize_acqf(
        acq_function=acquisition_function,
        bounds=bounds,
        q=batch_size,
        num_restarts=10,
        raw_samples=50,
    )

    # Visualization
    if plot and input_dim == 1:
        test_X = torch.linspace(bounds[0, 0], bounds[1, 0], steps=500).to(device, dtype)
        posterior = surrogate_model.posterior(test_X.unsqueeze(-1))
        pred_mean = posterior.mean.detach().cpu().numpy().flatten()
        pred_std = torch.sqrt(posterior.variance).detach().cpu().numpy().flatten()

        # Compute credible intervals
        lower_bound = pred_mean - 2 * pred_std
        upper_bound = pred_mean + 2 * pred_std

        # Convert data for plotting
        train_X_np = train_X.cpu().numpy().flatten()
        train_Y_np = train_Y.cpu().numpy().flatten()
        test_X_np = test_X.cpu().numpy().flatten()

        # Plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=train_X_np, y=train_Y_np, mode='markers', name='Training Data'))
        fig.add_trace(go.Scatter(x=test_X_np, y=pred_mean, mode='lines', name='Prediction'))
        fig.add_trace(go.Scatter(x=test_X_np, y=upper_bound, mode='lines', line=dict(dash='dot'), name='Upper CI'))
        fig.add_trace(go.Scatter(x=test_X_np, y=lower_bound, mode='lines', line=dict(dash='dot'), name='Lower CI'))
        fig.update_layout(
            title="Laplace BNN Fit with Credible Intervals",
            xaxis_title="Input X",
            yaxis_title="Output Y",
        )
        fig.show()

    return candidates

# -- VBLA ---------------------------------------------------
def vbla_sampler():
    pass


# -- GP ---------------------------------------------------
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood

def gp_sampler(
    train_X: torch.Tensor,
    train_Y: torch.Tensor,
    bounds: torch.Tensor,
    batch_size: int = 1,
    mc_acqf: Callable = qLogExpectedImprovement,
    device: torch.device = torch.device("cpu"),
    model_param_path: str = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    r"""
    Sample candidates using qLogExpectedImprovement with a SingleTaskGP surrogate model.

    Args:
        train_X: A `n x d` tensor of normalized training points.
        train_Y: A `n x 1` tensor of training targets.
        bounds: A `2 x d` tensor of lower and upper bounds for each dimension (already normalized).
        batch_size: Number of candidates to sample.
        device: Torch device (default: "cpu").
        dtype: Torch data type (default: float64).

    Returns:
        A `batch_size x d` tensor of candidate points in the normalized space.
    """
    # Ensure input data is on the correct device and type
    train_X = train_X.to(device=device, dtype=dtype)
    train_Y = train_Y.to(device=device, dtype=dtype)

    # Define and train the SingleTaskGP model
    model = SingleTaskGP(train_X, train_Y)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    # Define the acquisition function
    acquisition_function = mc_acqf(
        model=model,
        best_f=train_Y.max(),
        sampler=SobolQMCNormalSampler(sample_shape=torch.Size([500])),
    )

    # Optimize the acquisition function
    candidates, _ = optimize_acqf(
        acq_function=acquisition_function,
        bounds=bounds,  # Assumed to be already normalized
        q=batch_size,
        num_restarts=10,
        raw_samples=50,
    )

    return candidates

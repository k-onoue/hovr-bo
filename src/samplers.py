from typing import Callable

import torch
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.sampling import SobolQMCNormalSampler


# -- Laplace BNN -------------------------------------------
from .laplace_bnn import LaplaceBNN

def laplace_sampler(
    train_X: torch.Tensor,
    train_Y: torch.Tensor,
    bounds: torch.Tensor,
    batch_size: int = 1,
    mc_acqf: Callable = qLogExpectedImprovement,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float64,
    **kwargs
) -> torch.Tensor:
    r"""
    Sample candidates using the Laplace acquisition function.

    Args:
        train_X: A `n x d` tensor of training points.
        train_Y: A `n x 1` tensor of training targets.
        bounds: A `2 x d` tensor of lower and upper bounds for each dimension.
        batch_size: Number of candidates to sample.

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
    surrogate_model.fit(train_X, train_Y)

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

    return candidates


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

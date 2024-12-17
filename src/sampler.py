from abc import ABC, abstractmethod
from typing import Callable, Literal, Optional, Tuple

import torch
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.sampling import SobolQMCNormalSampler
from botorch.utils.transforms import normalize, unnormalize, standardize

from laplace import LaplaceBNN


def laplace_sampler(
    train_X: torch.Tensor,
    train_Y: torch.Tensor,
    bounds: torch.Tensor,
    batch_size: int = 1,
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
        output_dim=output_dim
    )
    surrogate_model.fit(train_X, train_Y)

    acquisition_function = qLogExpectedImprovement(
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


# --- Wrapper -----------------------------------------------------------------
class Sampler(ABC):
    @abstractmethod
    def sample(self, bounds: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        pass
    

class IndependentSampler(Sampler):
    def __init__(
        self, 
        n_initial_eval, 
        bounds: torch.Tensor,
        sample_method: Literal["sobol", "random"] = "sobol",
        device=None
    ):
        self.n_initial_eval = n_initial_eval
        self.bounds = bounds
        self.sample_method = sample_method
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def sample(self) -> torch.Tensor:
        bounds = self.bounds
        device = self.device
        dtype = bounds.dtype
        dim = bounds.shape[1]
        num = self.n_initial_eval

        if self.sample_method == "sobol":
            sobol = torch.quasirandom.SobolEngine(dimension=dim)
            samples = sobol.draw(num).to(device=device, dtype=dtype)
            samples = bounds[0] + (bounds[1] - bounds[0]) * samples
        elif self.sample_method == "random": 
            samples = torch.rand(num, bounds.shape[1], device=device, dtype=dtype)
            samples = bounds[0] + (bounds[1] - bounds[0]) * samples
        else:
            raise ValueError(f"Unknown sample method: {self.sample_method}")
        
        return samples


class RelativeSampler(Sampler):
    def __init__(
        self,
        train_X: torch.Tensor,
        train_Y: torch.Tensor,
        sampler: Callable,
        bounds: torch.Tensor,
        batch_size: int = 1, 
        dtype: torch.dtype = None,
        device: torch.device = None,
        **kwargs
    ):
        self.bounds = bounds
        self.batch_size = batch_size
        self.dtype = dtype or bounds.dtype
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Normalize inputs and standardize outputs
        self.train_X = normalize(
            train_X,
            bounds=bounds
        )
        self.train_Y = standardize(train_Y)
        
        # Store statistics for inverse transform
        self.Y_mean = self.train_Y.mean()
        self.Y_std = self.train_Y.std()

        self.sampler = sampler
        self.sampler_kwargs = kwargs
    
    def sample(self) -> torch.Tensor:
        # Sample in normalized space
        normalized_bounds = torch.stack([
            torch.zeros(self.bounds.shape[1], device=self.device, dtype=self.dtype),
            torch.ones(self.bounds.shape[1], device=self.device, dtype=self.dtype)
        ])
        
        normalized_candidates = self.sampler(
            train_X=self.train_X,
            train_Y=self.train_Y,
            bounds=normalized_bounds,
            batch_size=self.batch_size,
            dtype=self.dtype,
            device=self.device,
            **self.sampler_kwargs
        )
        
        # Transform back to original space
        candidates = unnormalize(
            normalized_candidates,
            bounds=self.bounds
        )
        return candidates
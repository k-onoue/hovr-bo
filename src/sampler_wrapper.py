from abc import ABC, abstractmethod
from typing import Callable, Literal, Optional, Tuple

import torch
from botorch.utils.transforms import normalize, unnormalize, standardize


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
        dtype=None,
        device=None,
    ):
        self.n_initial_eval = n_initial_eval
        self.bounds = bounds
        self.sample_method = sample_method
        self.dtype = dtype or bounds.dtype
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def sample(self) -> torch.Tensor:
        bounds = self.bounds
        device = self.device
        dtype = self.dtype
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
        acqf: Callable,
        bounds: torch.Tensor,
        batch_size: int = 1, 
        dtype: torch.dtype = None,
        device: torch.device = None,
        model_param_path: Optional[str] = None,
        **kwargs
    ):
        self.model_param_path = model_param_path

        self.bounds = bounds
        self.batch_size = batch_size
        self.dtype = dtype or bounds.dtype
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.sampler = sampler
        self.sampler_kwargs = kwargs
        self.acqf = acqf

        self.saved_model_state = None

        self.set_train_data(train_X, train_Y)

    def set_train_data(self, train_X: torch.Tensor, train_Y: torch.Tensor):
        # Normalize inputs and standardize outputs
        self.train_X = normalize(
            train_X,
            bounds=self.bounds
        )
        self.train_Y = standardize(train_Y)

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
            mc_acqf=self.acqf,
            dtype=self.dtype,
            device=self.device,
            model_param_path=self.model_param_path,
            **self.sampler_kwargs
        )
        
        # Transform back to original space
        candidates = unnormalize(
            normalized_candidates,
            bounds=self.bounds
        )
        
        return candidates


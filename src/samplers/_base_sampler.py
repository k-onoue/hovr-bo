from abc import ABC, abstractmethod
from typing import Literal, Optional, Tuple

import torch
from botorch.utils.transforms import normalize, unnormalize, standardize


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

        self.train_X = None
        self.train_Y = None

    def set_train_data(self, train_X: torch.Tensor, train_Y: torch.Tensor):
        # Normalize inputs and standardize outputs
        self.train_X = normalize(
            train_X,
            bounds=self.bounds
        )
        self.train_Y = standardize(train_Y)

    def _pre_process(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        train_X = self.train_X.to(device=self.device, dtype=self.dtype)
        train_Y = self.train_Y.to(device=self.device, dtype=self.dtype)

        # Use normalized bounds [0,1] since inputs are normalized
        norm_bounds = torch.tensor([[0.] * self.bounds.shape[1], 
                                  [1.] * self.bounds.shape[1]], 
                                 device=self.device,
                                 dtype=self.dtype)
        
        return train_X, train_Y, norm_bounds
    
    def _post_process(self, samples: torch.Tensor) -> torch.Tensor:
        return unnormalize(
            samples,
            bounds=self.bounds
        )

    @abstractmethod
    def _sample(self, train_X: torch.Tensor, train_Y: torch.Tensor, bounds: torch.Tensor) -> torch.Tensor:
        """Core sampling logic to be implemented by subclasses.
        
        Args:
            train_X: Normalized training inputs
            train_Y: Standardized training outputs  
            bounds: Normalized bounds tensor
            
        Returns:
            Normalized candidate points
        """
        pass

    def sample(self) -> torch.Tensor:
        """Full sampling pipeline with pre/post processing."""
        train_X, train_Y, norm_bounds = self._pre_process()
        candidates = self._sample(train_X, train_Y, norm_bounds)
        return self._post_process(candidates)
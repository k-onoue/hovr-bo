from functools import partial

import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.optim import optimize_acqf
from botorch.sampling import SobolQMCNormalSampler
from gpytorch.mlls import ExactMarginalLogLikelihood

from ._base_sampler import RelativeSampler
from ._utils import get_acquisition_function


class GPSampler(RelativeSampler):
    def __init__(
        self,
        bounds: torch.Tensor,
        batch_size: int = 1,
        dtype: torch.dtype = torch.float64,
        device: torch.device = torch.device("cpu"),
        acqf_name: str = "log_ei",
        **kwargs
    ) -> None:
        """Initialize GP-based sampler.
        
        Args:
            bounds: Original bounds for the input space
            batch_size: Number of points to sample
            dtype: Torch data type
            device: Torch device
            acqf_name: Name of acquisition function
        """
        super().__init__(
            bounds=bounds,
            batch_size=batch_size,
            dtype=dtype,
            device=device,
            acqf=acqf_name,
            **kwargs
        )
        
        self.surrogate = SingleTaskGP
        self.acqf_name = acqf_name
    
    def _initialize_acquisition(
        self,
        train_X: torch.Tensor,
        train_Y: torch.Tensor,
        model: SingleTaskGP,    
    ) -> callable:
        acqf_class = get_acquisition_function(self.acqf_name)

        base_acqf = partial(
            acqf_class, 
            model=model, 
            sampler=SobolQMCNormalSampler(sample_shape=torch.Size([1024]))
        )

        if self.acqf_name == "log_ei":
            acquisition_function = base_acqf(best_f=train_Y.max())
        elif self.acqf_name == "log_nei":
            acquisition_function = base_acqf(X_baseline=train_X)
        elif self.acqf_name == "ucb":
            acquisition_function = base_acqf(beta=0.1)
        else:
            raise NotImplementedError(f"Acquisition function '{self.acqf_name}' is not supported.")

        return acquisition_function
    
    def _sample(
        self, 
        train_X: torch.Tensor, 
        train_Y: torch.Tensor, 
        bounds: torch.Tensor
    ) -> torch.Tensor:

        model = self.surrogate(train_X, train_Y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)

        candidates, _ = optimize_acqf(
            acq_function=self._initialize_acquisition(train_X, train_Y, model),
            bounds=bounds,
            q=self.batch_size,
            num_restarts=10,
            raw_samples=50,
        )
        
        return candidates
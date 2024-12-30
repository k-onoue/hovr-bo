import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.optim import optimize_acqf
from botorch.sampling import SobolQMCNormalSampler
from botorch.utils.transforms import unnormalize
from gpytorch.mlls import ExactMarginalLogLikelihood

from ._base_sampler import RelativeSampler
from ._utils import get_acquisition_function


class GPSampler(RelativeSampler):
    def __init__(
        self,
        train_X: torch.Tensor,
        train_Y: torch.Tensor,
        bounds: torch.Tensor,
        batch_size: int = 1,
        dtype: torch.dtype = torch.float64,
        device: torch.device = torch.device("cpu"),
        acqf_name: str = "log_ei",
        **kwargs
    ) -> None:
        """Initialize GP-based sampler.
        
        Args:
            train_X: Training inputs
            train_Y: Training outputs
            bounds: Original bounds for the input space
            batch_size: Number of points to sample
            dtype: Torch data type
            device: Torch device
            acqf_name: Name of acquisition function
        """
        super().__init__(
            train_X=train_X,
            train_Y=train_Y, 
            bounds=bounds,
            batch_size=batch_size,
            dtype=dtype,
            device=device,
            acqf=acqf_name,
            **kwargs
        )
        
        self.surrogate = SingleTaskGP
        self.acquisition = get_acquisition_function(acqf_name)

    def _sample(self, train_X: torch.Tensor, train_Y: torch.Tensor, bounds: torch.Tensor) -> torch.Tensor:
        model = self.surrogate(train_X, train_Y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)

        acquisition_function = self.acquisition(
            model=model,
            best_f=train_Y.max(),
            sampler=SobolQMCNormalSampler(sample_shape=torch.Size([500])),
        )

        candidates, _ = optimize_acqf(
            acq_function=acquisition_function,
            bounds=bounds,
            q=self.batch_size,
            num_restarts=10,
            raw_samples=50,
        )
        
        return candidates
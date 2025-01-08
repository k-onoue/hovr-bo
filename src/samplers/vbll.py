import logging
from functools import partial

import torch
from botorch.optim import optimize_acqf
from botorch.sampling import SobolQMCNormalSampler

from ._base_sampler import RelativeSampler
from .models.vbll import VBLLModel
from ._utils import get_acquisition_function


class LastVBSampler(RelativeSampler):

    def __init__(
        self,
        bounds: torch.Tensor,
        batch_size: int = 1,
        dtype: torch.dtype = torch.float64,
        device: torch.device = torch.device("cpu"),
        acqf_name: str = "log_ei",
        **kwargs
    ) -> None:
        """Initialize VBLL (Variational Bayesian Last Layer) MLP-based sampler.
        
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
        
        self.surrogate = self._initialize_vbll(
            kwargs.get("surrogate_args", {})
        )

        self.acqf_name = acqf_name

        self.optim_config = kwargs.get("optim_args", {})

        logging.info(kwargs.get("surrogate_args", {}))
        logging.info(self.optim_config)

    def _initialize_vbll(self, surrogate_args: dict) -> VBLLModel:

        return VBLLModel(
            dimensions=surrogate_args.get("dimensions", [128, 128, 128]),
            activation=surrogate_args.get("activation", "tanh"),
            input_dim=self.bounds.shape[1],
            output_dim=1,
            dtype=self.dtype,
            device=self.device,
            reg_weight=surrogate_args.get("reg_weight", 1.0),
            parameterization=surrogate_args.get("parameterization", "dense"),
            prior_scale=surrogate_args.get("prior_scale", 1.0),
            wishart_scale=surrogate_args.get("wishart_scale", 0.1),
        )

    def _initialize_acquisition(
        self,
        train_X: torch.Tensor,
        train_Y: torch.Tensor    
    ) -> callable:
        acqf_class = get_acquisition_function(self.acqf_name)

        base_acqf = partial(
            acqf_class, 
            model=self.surrogate, 
            sampler=SobolQMCNormalSampler(sample_shape=torch.Size([1024]))
        )

        if self.acqf_name == "log_ei":
            acquisition_function = base_acqf(best_f=train_Y.max())
        elif self.acqf_name == "log_nei":
            acquisition_function = base_acqf(X_baseline=train_X)
        elif self.acqf_name == "ucb":
            acquisition_function = base_acqf(beta=2)
        else:
            raise NotImplementedError(f"Acquisition function '{self.acqf_name}' is not supported.")

        return acquisition_function
    
    def _sample(
        self, 
        train_X: torch.Tensor, 
        train_Y: torch.Tensor, 
        bounds: torch.Tensor
    ) -> torch.Tensor:

        self.surrogate.fit(train_X, train_Y, config=self.optim_config)

        candidates, _ = optimize_acqf(
            acq_function=self._initialize_acquisition(train_X, train_Y),
            bounds=bounds,
            q=self.batch_size,
            num_restarts=10,
            raw_samples=512,
        )
        
        return candidates
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
            kwargs.get("model_args", {})
        )
        self.acquisition = get_acquisition_function(acqf_name)

        self.optim_config = kwargs.get("optim_args", {})

    def _initialize_vbll(self, model_args: dict) -> VBLLModel:

        model = VBLLModel(
            dimensions=model_args.get("dimensions", [128, 128, 128]),
            activation=model_args.get("activation", "tanh"),
            input_dim=self.bounds.shape[1],
            output_dim=1,
            dtype=self.dtype,
            device=self.device,
            reg_weight=model_args.get("reg_weight", 1.0),
            parameterization=model_args.get("parameterization", "dense"),
            prior_scale=model_args.get("prior_scale", 1.0),
            wishart_scale=model_args.get("wishart_scale", 0.1),
        )

        return model

    def _sample(self, train_X: torch.Tensor, train_Y: torch.Tensor, bounds: torch.Tensor) -> torch.Tensor:

        self.surrogate.fit(train_X, train_Y, config=self.optim_config)

        acquisition_function = self.acquisition(
            model=self.surrogate,
            best_f=train_Y.max(),
            sampler=SobolQMCNormalSampler(sample_shape=torch.Size([1024])),
        )

        candidates, _ = optimize_acqf(
            acq_function=acquisition_function,
            bounds=bounds,
            q=self.batch_size,
            num_restarts=10,
            raw_samples=50,
        )
        
        return candidates
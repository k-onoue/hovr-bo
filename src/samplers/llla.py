import logging
from functools import partial

import torch
from botorch.optim import optimize_acqf
from botorch.sampling import SobolQMCNormalSampler

from ._base_sampler import RelativeSampler
from .models.llla import LaplaceModel
from ._utils import get_acquisition_function


class LastLaplaceL2Sampler(RelativeSampler):
    
    def __init__(
        self,
        bounds: torch.Tensor,
        batch_size: int = 1,
        dtype: torch.dtype = torch.float64,
        device: torch.device = torch.device("cpu"),
        acqf_name: str = "log_ei",
        **kwargs
    ) -> None:
        """Initialize Laplace L2-based sampler.
        
        Args:
            bounds: Original bounds for the input space
            batch_size: Number of points to sample
            dtype: Torch data type
            device: Torch device
            acqf_name: Name of acquisition function
        
        config = {
            # Data loading
            "batch_size": 32,          # Batch size for training
            "val_split": 0.2,          # Validation split ratio
            "min_train_size": 5,       # Minimum dataset size for validation split

            # Optimization
            "optimizer": torch.optim.Adam,  # Optimizer class
            "lr": 1e-3,                    # Learning rate
            "weight_decay_hidden": 0,      # Weight decay for non-output layer weights
            "weight_decay_last": 0,        # Weight decay for output layer weights
            "epochs": 1000,                # Number of training epochs

            # Early Stopping
            "patience": 20,           # Number of epochs to wait for improvement
            "verbose": True,         # Print early stopping messages
            "delta": 0,              # Minimum change to qualify as improvement

            # Laplace approximation
            "hessian_structure": "full",
            "prior_precision": 1e-2,  # Too small prior precision can lead to numerical instability
            "sigma_noise": 1e-1,
            "temperature": 1,
        }
        """
        super().__init__(
            bounds=bounds,
            batch_size=batch_size,
            dtype=dtype,
            device=device,
            acqf=acqf_name,
            **kwargs
        )
        
        self.surrogate = self._initialize_laplace(
            kwargs.get("surrogate_args", {})
        )

        self.acqf_name = acqf_name

        self.optim_config = kwargs.get("optim_args", {})
        self.optim_config["loss_coeffs"] = {
            "mse": 1.0,
            "trim": 0.0,
            "hovr": 0.0,
        }

        logging.info(kwargs.get("surrogate_args", {}))
        logging.info(self.optim_config)

    def _initialize_laplace(self, surrogate_args: dict) -> LaplaceModel:

        return LaplaceModel(
            dimensions=surrogate_args.get("dimensions", [128, 128, 128]),
            activation=surrogate_args.get("activation", "tanh"),
            input_dim=self.bounds.shape[1],
            output_dim=1,
            dtype=self.dtype,
            device=self.device
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
            acquisition_function = base_acqf(X_baseline=train_X, cache_root=False)
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


class LastLaplaceARTLSampler(RelativeSampler):
    
    def __init__(
        self,
        bounds: torch.Tensor,
        batch_size: int = 1,
        dtype: torch.dtype = torch.float64,
        device: torch.device = torch.device("cpu"),
        acqf_name: str = "log_ei",
        **kwargs
    ) -> None:
        """Initialize Laplace ARTL-based sampler.
        
        Args:
            bounds: Original bounds for the input space
            batch_size: Number of points to sample
            dtype: Torch data type
            device: Torch device
            acqf_name: Name of acquisition function
        
        config = {
            # Data loading
            "batch_size": 32,          # Batch size for training
            "val_split": 0.2,          # Validation split ratio
            "min_train_size": 5,       # Minimum dataset size for validation split

            # Optimization
            "optimizer": torch.optim.Adam,  # Optimizer class
            "lr": 1e-3,                    # Learning rate
            "weight_decay_hidden": 0,      # Weight decay for non-output layer weights
            "weight_decay_last": 0,        # Weight decay for output layer weights
            "epochs": 1000,                # Number of training epochs

            # Loss coefficients
            "loss_coeffs": {
                "mse": 1,                 # MSE loss coefficient
                "trim": 1e-1,                # Trimmed loss coefficient
                "hovr": 1e-3,                # HOVR loss coefficient
            }

            # Loss parameters
            loss_params = {
                "h": None,                # Number of points to keep after trimming
                "k": (1, 2),              # Tuple of derivative orders
                "q": 2,                   # Exponent in HOVR
                "M": 10,                  # Number of random points
            }

            # Early Stopping
            "patience": 20,           # Number of epochs to wait for improvement
            "verbose": True,         # Print early stopping messages
            "delta": 0,              # Minimum change to qualify as improvement

            # Laplace approximation
            "hessian_structure": "full",
            "prior_precision": 1e-2,  # Too small prior precision can lead to numerical instability
            "sigma_noise": 1e-1,
            "temperature": 1,
        }
        """
        super().__init__(
            bounds=bounds,
            batch_size=batch_size,
            dtype=dtype,
            device=device,
            acqf=acqf_name,
            **kwargs
        )

        # Extract optimization configuration
        self.optim_config = kwargs.get("optim_args", {})

        self._validate_config()

        # Initialize surrogate model
        self.surrogate = self._initialize_laplace(
            kwargs.get("surrogate_args", {})
        )

        # Initialize acquisition function
        self.acqf_name = acqf_name

        logging.info(kwargs.get("surrogate_args", {}))
        logging.info(self.optim_config)

    def _validate_config(self):
        """Validate the optimization configuration."""
        loss_coeffs = self.optim_config.get("loss_coeffs", {})
        
        if not loss_coeffs:
            raise ValueError("'loss_coeffs' must not be empty.")

        trim_coeff = loss_coeffs.get("trim", 0)
        hovr_coeff = loss_coeffs.get("hovr", 0)

        if trim_coeff <= 0 and hovr_coeff <= 0:
            raise ValueError(
                "At least one of 'trim' or 'hovr' coefficients must be greater than 0."
            )

    def _initialize_laplace(self, surrogate_args: dict) -> LaplaceModel:

        return LaplaceModel(
            dimensions=surrogate_args.get("dimensions", [128, 128, 128]),
            activation=surrogate_args.get("activation", "tanh"),
            input_dim=self.bounds.shape[1],
            output_dim=1,
            dtype=self.dtype,
            device=self.device
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
            acquisition_function = base_acqf(X_baseline=train_X, cache_root=False)
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

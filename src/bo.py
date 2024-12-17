from abc import ABC, abstractmethod
import torch
from botorch.sampling import SobolQMCNormalSampler
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.optim import optimize_acqf
from typing import Optional, Tuple
# from _src import LaplaceBNN
from laplace import LaplaceBNN



class Sampler(ABC):
    @abstractmethod
    def sample(self, bounds: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        pass

class IndependentSampler(Sampler):
    def __init__(self, n_samples: int, sample_method: str = "sobol", device=None):
        self.n_samples = n_samples
        self.sample_method = sample_method
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def sample(self, bounds: torch.Tensor) -> Tuple[torch.Tensor, None]:
        if self.sample_method == "sobol":
            sobol = torch.quasirandom.SobolEngine(dimension=bounds.shape[1])
            samples = sobol.draw(self.n_samples).to(device=self.device, dtype=bounds.dtype)
            samples = bounds[0] + (bounds[1] - bounds[0]) * samples
        else:  # random
            samples = torch.rand(self.n_samples, bounds.shape[1], device=self.device, dtype=bounds.dtype)
            samples = bounds[0] + (bounds[1] - bounds[0]) * samples
        return samples, None

class RelativeSampler(Sampler):
    def __init__(self, surrogate_model, n_samples: int = 500, device=None):
        self.surrogate = surrogate_model
        self.n_samples = n_samples
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.qmc_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([n_samples]))
    
    def sample(self, bounds: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Create acquisition function using surrogate model
        acq_function = qLogNoisyExpectedImprovement(
            model=self.surrogate,
            X_baseline=self.surrogate.train_inputs[0],
            sampler=self.qmc_sampler,
        )
        
        # Optimize acquisition function
        candidates, acq_values = optimize_acqf(
            acq_function=acq_function,
            bounds=bounds,
            q=1,
            num_restarts=10,
            raw_samples=self.n_samples,
        )
        
        return candidates, acq_values

class BayesianOptimization:
    def __init__(self, 
                 model_class, 
                 model_args,
                 independent_sampler: Optional[IndependentSampler] = None,
                 device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], device=self.device, dtype=torch.float64)
        
        # Initialize independent sampler
        self.independent_sampler = independent_sampler or IndependentSampler(n_samples=5)
        
        # Get initial samples
        self.train_x, _ = self.independent_sampler.sample(self.bounds)
        self.train_y = self._evaluate_objective(self.train_x)
        
        # Initialize surrogate model
        self.model = model_class(model_args, input_dim=2, output_dim=1, device=self.device)
        self.model.fit(self.train_x, self.train_y)
        
        # Initialize relative sampler with surrogate
        self.relative_sampler = RelativeSampler(surrogate_model=self.model)

    def _evaluate_objective(self, x):
        return self.branin(x).unsqueeze(-1)

    def optimize(self, iterations=10):
        for iteration in range(iterations):
            # Sample using relative sampler
            new_x, _ = self.relative_sampler.sample(self.bounds)
            new_y = self._evaluate_objective(new_x)
            
            # Update data and refit model
            self.train_x = torch.cat([self.train_x, new_x])
            self.train_y = torch.cat([self.train_y, new_y])
            self.model.fit(self.train_x, self.train_y)
            
            print(f"Iteration {iteration + 1}")
            print(f"New point: {new_x}")
            print(f"Objective value: {new_y}")



if __name__ == "__main__":

    model_args = {
        "regnet_dims": [10, 10],
        "regnet_activation": "relu",
        "prior_var": 1.0,
        "noise_var": 1e-2,
        "iterative": False,
    }

    bo = BayesianOptimization(
        model_class=LaplaceBNN,
        model_args=model_args,
    )
    bo.optimize(iterations=10)
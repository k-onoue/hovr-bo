import torch

from _src import BayesianOptimization
from _src import gp_sampler, laplace_sampler
from _src import SyntheticSine



if __name__ == "__main__":


    noise_std = 2
    outlier_prob = 0
    outlier_scale = 30
    objective_function = SyntheticSine(
        noise_std=noise_std,
        outlier_prob=outlier_prob,
        outlier_scale=outlier_scale
    )

    bo = BayesianOptimization(
        objective_function=objective_function,
        sampler=gp_sampler,
        n_initial_eval=5,
        n_iter=30,
        batch_size=2,
        is_maximize=False,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )

    bo.run()
    bo.report()
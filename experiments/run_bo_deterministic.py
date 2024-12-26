import argparse
import os

import torch
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.acquisition.logei import qLogNoisyExpectedImprovement

from _src import BayesianOptimization
from _src import gp_sampler, laplace_sampler
from _src import SyntheticSine, BraninFoo
from _src import set_logger



class Experiment:
    def __init__(self, settings):
        self.settings = settings

    def _unpack_basic_settings(self):
        return (
            self.settings["n_initial_eval"], 
            self.settings["n_iter"], 
            self.settings["batch_size"],
            self.settings["is_maximize"],
            self.settings["device"],
            self.settings["dtype"],
            self.settings.get("seed", None),
        )
    
    def _unpack_objective_settings(self):
        return (
            self.settings["objective"]["function"],
            self.settings["objective"]["noise_std"],
            self.settings["objective"]["outlier_prob"],
            self.settings["objective"]["outlier_scale"],
            self.settings["objective"]["outlier_std"],
        )
    def _unpack_sampler_settings(self):
        return (
            self.settings["sampler"]["function"],
            self.settings["sampler"]["acqf"],
            self.settings["sampler"].get("sampler_args", None),
        )

    def run(self):
        
        n_initial_eval, n_iter, batch_size, is_maximize, device, dtype, seed = self._unpack_basic_settings()
        objective_function, noise_std, outlier_prob, outlier_scale, outlier_std = self._unpack_objective_settings()

        sampler, acqf, sampler_args = self._unpack_sampler_settings()


        
        objective_function = objective_function(
            noise_std=noise_std,
            outlier_prob=outlier_prob,
            outlier_scale=outlier_scale,
            outlier_std=outlier_std,
        )

        bo = BayesianOptimization(
            objective_function=objective_function,
            sampler=sampler,
            n_initial_eval=n_initial_eval,
            n_iter=n_iter,
            batch_size=batch_size,
            is_maximize=is_maximize,
            device=device,
            dtype=dtype,
            acqf=acqf,
            seed=seed,
            sampler_args=sampler_args,
        )

        bo.run()
        bo.report()


def parse_args():
    parser = argparse.ArgumentParser(description="Bayesian Optimization with BNNs")
    # Basic parameters
    parser.add_argument("--timestamp", type=str, help="Timestamp for the experiment")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--obj", type=str, default="synthetic", help="Objective function")
  

def get_objective_function(name):
    if name == "synthetic":
        return SyntheticSine
    elif name == "branin":
        return BraninFoo
    else:
        raise ValueError(f"Objective function {name} not recognized.")
    

def get_acquisition_function(name):
    if name == "log_ei":
        return qLogExpectedImprovement
    elif name == "log_nei":
        return qLogNoisyExpectedImprovement
    
    else:
        raise ValueError(f"Acquisition function {name} not recognized.")



if __name__ == "__main__":
    base_script_name = os.path.splitext(__file__.split("/")[-1])[0]
    args = parse_args()

    timestamp = args.timestamp
    results_dir = os.path.join("results", timestamp)
    os.makedirs(results_dir, exist_ok=True)

    log_filename_base = f"{base_script_name}_{args.obj}_{args.seed}"

    log_filepath = set_logger(log_filename_base, results_dir)

    settings = {
        "n_initial_eval": 10,
        "n_iter": 10,
        "batch_size": 10,
        "is_maximize": False,
        "device": torch.device("cpu"),
        "dtype": torch.float64,
        "objective": {
            "name": "synthetic_sine",
            "function": SyntheticSine,
            "noise_std": 1.0,
            "outlier_prob": 0.0,
            "outlier_scale": 10,
            "outlier_std": 1.0,
        },
        "sampler": {
            "name": "laplace",
            "function": laplace_sampler,
            "acqf": qLogExpectedImprovement,
            "sampler_args": {
                "loss_params": {
                    "n_epochs": 1000,
                    "lr": 1e-2,
                    "weight_decay": 0,
                    "artl_weight": 1e-3,
                    "lambd": 0,
                    "k": (1, 2, 3),
                    "q": 1,
                    "M": 10,
                },
            },
        }
    }

    experiment = Experiment(settings)
    experiment.run()


    # setting_list = [

        # { # not good
        #     "regnet_dims": [128, 128, 128],
        #     "regnet_activation": "tanh",
        #     "prior_var": 1.0,
        #     "noise_var": 1.0,
        #     "iterative": True,
        #     "loss_params": {
        #         "n_epochs": 1000,
        #         "lr": 1e-2,
        #         "weight_decay": 0,
        #         "momentum": 0.9,
        #         "artl_weight": 1,
        #         "lambd": 1e-3,
        #         "k": (1, 2, 3,),
        #         "q": 1,
        #         "M": 10,
        #     },
        # },
        # { # good
        #     "regnet_dims": [128, 128, 128],
        #     "regnet_activation": "tanh",
        #     "prior_var": 1.0,
        #     "noise_var": 1.0,
        #     "iterative": True,
        #     "loss_params": {
        #         "n_epochs": 1000,
        #         "lr": 1e-2,
        #         "weight_decay": 0,
        #         "momentum": 0.9,
        #         "artl_weight": 1e-3,
        #         "lambd": 1e-3,
        #         "k": (3,),
        #         "q": 1,
        #         "M": 10,
        #     },
        # },
    #     { # no regularization
    #         "regnet_dims": [128, 128, 128],
    #         "regnet_activation": "tanh",
    #         "prior_var": 1.0,
    #         "noise_var": 1.0,
    #         "iterative": True,
    #         "loss_params": {
    #             "n_epochs": 1000,
    #             "lr": 1e-2,
    #             "weight_decay": 0,
    #             "momentum": 0.9,
    #             "artl_weight": 0,
    #             "lambd": 1e-3,
    #             "k": (3,),
    #             "q": 1,
    #             "M": 10,
    #         },
    #     },
    #     { # l2 regularization
    #         "regnet_dims": [128, 128, 128],
    #         "regnet_activation": "tanh",
    #         "prior_var": 1.0,
    #         "noise_var": 1.0,
    #         "iterative": True,
    #         "loss_params": {
    #             "n_epochs": 1000,
    #             "lr": 1e-2,
    #             "weight_decay": 1e-3,
    #             "momentum": 0.9,
    #             "artl_weight": 0,
    #             "lambd": 1e-3,
    #             "k": (3,),
    #             "q": 1,
    #             "M": 10,
    #         },
    #     },
    # ]
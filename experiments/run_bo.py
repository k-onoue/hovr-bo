import torch
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.acquisition.logei import qLogNoisyExpectedImprovement

from _src import BayesianOptimization
from _src import gp_sampler, laplace_sampler
from _src import SyntheticSine



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
        
        n_initial_eval, n_iter, batch_size, is_maximize, device, dtype = self._unpack_basic_settings()
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
            sampler_args=sampler_args,
        )

        bo.run()
        bo.report()


if __name__ == "__main__":

    setting_list = [
        # {
        #     "regnet_dims": [128, 128, 128],
        #     "regnet_activation": "tanh",
        #     "prior_var": 1.0,
        #     "noise_var": 1.0,
        #     "iterative": True,
        #     "loss_params": {
        #         "n_epochs": 1000,
        #         "lr": 1e-2,
        #         "weight_decay": 0,
        #         "artl_weight": 1e-3,
        #         "lambd": 0,
        #         "k": (0,),
        #         "q": 0,
        #         "M": 10,
        #     },
        # },
        # {
        #     "regnet_dims": [128, 128, 128],
        #     "regnet_activation": "tanh",
        #     "prior_var": 1.0,
        #     "noise_var": 1.0,
        #     "iterative": True,
        #     "loss_params": {
        #         "n_epochs": 1000,
        #         "lr": 1e-2,
        #         "weight_decay": 0,
        #         "artl_weight": 1e-3,
        #         "lambd": 1e-3,
        #         "k": (1, 2, 3),
        #         "q": 1,
        #         "M": 10,
        #     },
        # },
        # {
        #     "regnet_dims": [128, 128, 128],
        #     "regnet_activation": "tanh",
        #     "prior_var": 1.0,
        #     "noise_var": 1.0,
        #     "iterative": True,
        #     "loss_params": {
        #         "n_epochs": 1000,
        #         "lr": 1e-2,
        #         "weight_decay": 0,
        #         "artl_weight": 1e-3,
        #         "lambd": 1e-3,
        #         "k": (3,),
        #         "q": 1,
        #         "M": 10,
        #     },
        # },
        # {
        #     "regnet_dims": [128, 128, 128],
        #     "regnet_activation": "tanh",
        #     "prior_var": 1.0,
        #     "noise_var": 1.0,
        #     "iterative": True,
        #     "loss_params": {
        #         "n_epochs": 1000,
        #         "lr": 1e-2,
        #         "weight_decay": 1e-2,
        #         "artl_weight": 1e-3,
        #         "lambd": 0,
        #         "k": (0,),
        #         "q": 0,
        #         "M": 10,
        #     },
        # },
        {
            "regnet_dims": [128, 128, 128],
            "regnet_activation": "tanh",
            "prior_var": 1.0,
            "noise_var": 1.0,
            "iterative": True,
            "loss_params": {
                "n_epochs": 1000,
                "lr": 1e-2,
                "weight_decay": 1e-2,
                "momentum": 0,
                "artl_weight": 0,
                "lambd": 0,
                "k": (0,),
                "q": 0,
                "M": 10,
            },
        },
        {
            "regnet_dims": [128, 128, 128],
            "regnet_activation": "tanh",
            "prior_var": 1.0,
            "noise_var": 1.0,
            "iterative": True,
            "loss_params": {
                "n_epochs": 1000,
                "lr": 1e-2,
                "weight_decay": 1e-3,
                "momentum": 0,
                "artl_weight": 0,
                "lambd": 0,
                "k": (0,),
                "q": 0,
                "M": 10,
            },
        },
    ]

    for setting in setting_list:

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
                "sampler_args": setting,
            }
        }

        experiment = Experiment(settings)
        experiment.run()

    # settings = {
    #     "n_initial_eval": 10,
    #     "n_iter": 10,
    #     "batch_size": 10,
    #     "is_maximize": False,
    #     "device": torch.device("cpu"),
    #     "dtype": torch.float64,
    #     "objective": {
    #         "name": "synthetic_sine",
    #         "function": SyntheticSine,
    #         "noise_std": 1.0,
    #         "outlier_prob": 0.0,
    #         "outlier_scale": 10,
    #         "outlier_std": 1.0,
    #     },
    #     "sampler": {
    #         "name": "laplace",
    #         "function": laplace_sampler,
    #         "acqf": qLogExpectedImprovement,
    #         "sampler_args": {
    #             "regnet_dims": [128, 128, 128],
    #             "regnet_activation": "tanh",
    #             "prior_var": 1.0,
    #             "noise_var": 1.0,
    #             "iterative": True,
    #             "loss_params": {
    #                 "n_epochs": 1000,
    #                 "lr": 1e-2,
    #                 "weight_decay": 0,
    #                 "artl_weight": 1e-3,
    #                 "lambd": 0,
    #                 "k": (1, 2, 3),
    #                 "q": 1,
    #                 "M": 10,
    #             },
    #         },
    #     }
    # }

    # experiment = Experiment(settings)
    # experiment.run()
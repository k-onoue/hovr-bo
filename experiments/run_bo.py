import argparse
import json
import os

import torch

from _src import BayesianOptimization
from _src import set_logger
from _src import get_objective_function, get_sampler


class Experiment:
    def __init__(self, settings):
        self.settings = settings

    def _unpack_basic_settings(self):
        return (
            int(self.settings["n_initial_eval"]), 
            self.settings["initial_sample_method"],
            int(self.settings["n_iter"]), 
            int(self.settings["batch_size"]),
            bool(self.settings["is_maximize"]),
            torch.device(self.settings["device"]),
            getattr(torch, self.settings["dtype"]),
            int(self.settings.get("seed", 0)),
        )
    
    def _unpack_objective_settings(self):
        obj_settings = self.settings["objective"]
        return (
            get_objective_function(obj_settings["function"]),
            float(obj_settings["noise_std"]),
            float(obj_settings["outlier_prob"]),
            float(obj_settings["outlier_scale"]),
            float(obj_settings["outlier_std"]),
        )

    def _unpack_sampler_settings(self):
        sampler_settings = self.settings["sampler"]
        return (
            get_sampler(sampler_settings["surrogate"]),
            sampler_settings.get("acqf", "log_ei"),  # Default acquisition function if not specified
            sampler_settings.get("sampler_args", {}),
        )

    def run(self, save_dir: str):
        n_initial_eval, initial_sample_method, n_iter, batch_size, is_maximize, device, dtype, seed = self._unpack_basic_settings()
        objective_function, noise_std, outlier_prob, outlier_scale, outlier_std = self._unpack_objective_settings()

        sampler, acqf_name, sampler_args = self._unpack_sampler_settings()

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
            initial_sample_method=initial_sample_method,
            n_iter=n_iter,
            batch_size=batch_size,
            is_maximize=is_maximize,
            device=device,
            dtype=dtype,
            acqf_name=acqf_name,  # Updated parameter
            seed=seed,
            sampler_args=sampler_args,
        )

        bo.run()
        bo.report()

        _seed = seed
        _obj = self.settings["objective"]["function"]
        _sampler = self.settings["sampler"]["surrogate"]
        _acqf = self.settings["sampler"].get("acqf", "log_ei")

        filename = f"{_obj}_{_sampler}_{_acqf}_{_seed}.csv"
        filename = os.path.join(save_dir, filename)
        bo.history_df.to_csv(filename, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings", type=str, required=True, help="Path to settings JSON file")
    parser.add_argument("--timestamp", type=str, required=True, help="Timestamp for the experiment")
    parser.add_argument("--seed", type=int, default=0, help="Seed for the experiment")
    args = parser.parse_args()

    with open(args.settings, "r") as f:
        settings = json.load(f)

    experiment = Experiment(settings)
    base_script_name = os.path.splitext(os.path.basename(__file__))[0]
    results_dir = os.path.join("results", args.timestamp)
    os.makedirs(results_dir, exist_ok=True)
    experiment.run(save_dir=results_dir)
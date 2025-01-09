import argparse
import json
import logging
import os
import pandas as pd
import torch
from _src import BayesianOptimization, set_logger, get_objective_function, get_sampler


class Experiment:
    def __init__(self, settings, sampler_type):
        self.settings = settings
        self.sampler_type = sampler_type

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
        sampler_settings = self.settings["samplers"][self.sampler_type]
        return (
            get_sampler(self.sampler_type),
            sampler_settings.get("acqf", "log_ei"),
            sampler_settings.get("sampler_args", {}),
        )
    
    def run(self, save_dir: str):
        n_initial_eval, initial_sample_method, n_iter, batch_size, is_maximize, device, dtype, seed = self._unpack_basic_settings()
        objective_function, noise_std, outlier_prob, outlier_scale, outlier_std = self._unpack_objective_settings()
        sampler, acqf_name, sampler_args = self._unpack_sampler_settings()

        _seed = seed
        _obj = self.settings["objective"]["function"]
        filename_csv = f"{_obj}_{self.sampler_type}_{acqf_name}_{_seed}.csv"
        filepath_csv = os.path.join(save_dir, filename_csv)
        filename_png = f"{_obj}_{self.sampler_type}_{acqf_name}_{_seed}.png"
        filepath_png = os.path.join(save_dir, filename_png)

        os.makedirs(save_dir, exist_ok=True)

        if os.path.exists(filepath_csv):
            df = pd.read_csv(filepath_csv)
            completed_iters = len(df[df['sampler'] == 'relative']) // batch_size
            remaining_iters = n_iter - completed_iters
            
            if remaining_iters <= 0:
                logging.info("Previous run was completed")
                return
                
            n_iter = remaining_iters
            logging.info(f"Resuming optimization for {remaining_iters} iterations")

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
            acqf_name=acqf_name,
            seed=seed,
            sampler_args=sampler_args,
        )

        logging.info(f"Starting Bayesian Optimization with settings: {self.settings}")
        logging.info(f"Results will be saved to: {filepath_csv}")

        bo.run(save_path=filepath_csv)
        bo.report(save_path=filepath_png)
        logging.info(f"Optimization completed. Results saved to {filepath_csv} and {filepath_png}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--sampler_type", type=str, required=True, help="Type of sampler to use")
    parser.add_argument("--timestamp", type=str, required=True, help="Timestamp for the experiment")
    parser.add_argument("--seed", type=int, default=0, help="Seed for the experiment")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        settings = json.load(f)
    
    settings["seed"] = args.seed
    experiment = Experiment(settings, args.sampler_type)
    
    base_script_name = os.path.splitext(os.path.basename(__file__))[0]
    results_dir = os.path.join("results", args.timestamp)
    os.makedirs(results_dir, exist_ok=True)

    set_logger(base_script_name, results_dir)
    logging.info(f"Running experiment with settings: {settings}")
    logging.info(f"Using sampler: {args.sampler_type}")

    experiment.run(save_dir=results_dir)
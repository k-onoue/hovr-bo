import logging
import os
import random
from typing import Callable, Literal, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tabulate import tabulate
import torch

from .samplers._base_sampler import IndependentSampler, RelativeSampler


class BayesianOptimization:
    def __init__(
        self,
        objective_function: Callable,
        sampler: RelativeSampler,
        n_initial_eval: int = 5,
        initial_sample_method: str = "sobol",
        n_iter: int = 30,
        batch_size: int = 1,
        is_maximize: bool = False,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float64,
        acqf_name: str = "log_ei",
        seed: int = None,
        sampler_args: Optional[dict] = None,
    ):
        self.objective_function = objective_function
        self.sampler = sampler
        self.n_initial_eval = n_initial_eval
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.is_maximize = is_maximize
        self.device = device
        self.dtype = dtype
        self.seed = seed
        self._set_seed()
        
        self.bounds = torch.tensor(
            self.objective_function._bounds,
            device=self.device,
            dtype=self.dtype
        ).t()
        
        self.independent_sampler = IndependentSampler(
            seed=self.seed,
            n_initial_eval=n_initial_eval,
            bounds=self.bounds,
            sample_method=initial_sample_method,
            dtype=self.dtype,
            device=self.device,
        )
        
        self.relative_sampler = sampler(
            bounds=self.bounds,
            batch_size=batch_size,
            dtype=self.dtype,
            device=self.device,
            acqf_name=acqf_name,
            **(sampler_args or {})
        )
        
        self.X_all = None
        self.Y_all = None
        self.F_all = None
        self.history_df = None

    def _set_seed(self):
        if self.seed is not None:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)
            random.seed(self.seed)

    def load_existing_data(self, filepath: str):
        """Load data from previous run with proper type conversion."""
        df = pd.read_csv(filepath)
        n_dim = len([col for col in df.columns if col.startswith('x')])
        
        # Convert input columns to float
        X_cols = [f'x{i}' for i in range(n_dim)]
        X_data = df[X_cols].astype(float).to_numpy()
        Y_data = df['Y'].astype(float).to_numpy()
        F_data = df['F'].astype(float).to_numpy()
        
        # Convert to tensors
        self.X_all = torch.tensor(X_data, dtype=self.dtype, device=self.device).reshape(-1, n_dim)
        self.Y_all = torch.tensor(Y_data, dtype=self.dtype, device=self.device).reshape(-1, 1)
        self.F_all = torch.tensor(F_data, dtype=self.dtype, device=self.device).reshape(-1, 1)
        self.history_df = df
        
        # Return number of completed relative sampler iterations
        return len(df[df['sampler'] == 'relative'])

    def run(self, save_path: Optional[str] = None):
        if save_path and os.path.exists(save_path):
            n_completed = self.load_existing_data(save_path)
            logging.info(f"Resumed from {save_path} with {n_completed} completed evaluations")
        else:
            X_init = self.independent_sampler.sample()
            Y_init, F_init = self._evaluate(X_init)
            self.X_all = X_init
            self.Y_all = Y_init
            self.F_all = F_init
            self._record("independent", save_path=save_path)

        for i in range(self.n_iter):

            self.relative_sampler.set_train_data(
                train_X=self.X_all,
                train_Y=self.Y_all if self.is_maximize else -self.Y_all
            )

            X_new = self.relative_sampler.sample()
            Y_new, F_new = self._evaluate(X_new)
            
            self.X_all = torch.cat([self.X_all, X_new], dim=0)
            self.Y_all = torch.cat([self.Y_all, Y_new], dim=0)
            self.F_all = torch.cat([self.F_all, F_new], dim=0)
            
            self._record("relative", save_path=save_path)
            logging.info(f"Iteration {i+1}/{self.n_iter} completed.")

        logging.info("Optimization completed.")

    def _evaluate(self, X: torch.Tensor):
        Y = self.objective_function(X)
        F = self.objective_function.evaluate_true(X)
        return Y, F
    
    def _record(self, sampler_name: str, save_path: Optional[str] = None):
        X = self.X_all
        Y = self.Y_all.unsqueeze(-1) if self.Y_all.dim() == 1 else self.Y_all
        F = self.F_all.unsqueeze(-1) if self.F_all.dim() == 1 else self.F_all

        if self.is_maximize:
            y_best = Y.max()
            f_best = F.max()
        else:
            y_best = Y.min()
            f_best = F.min()

        # Convert to CPU for compatibility with Pandas
        X_np = X.cpu().numpy()
        Y_np = Y.cpu().numpy()
        F_np = F.cpu().numpy()
        y_best_np = y_best.cpu().numpy()
        f_best_np = f_best.cpu().numpy()

        if self.history_df is None:
            # Initialize history_df on the first iteration
            _ones = np.ones_like(Y_np)
            self.history_df = pd.DataFrame(
                np.concatenate([X_np, Y_np, F_np], axis=1),
                columns=[f"x{i}" for i in range(X.shape[1])] + ["Y", "F"]
            )
            self.history_df["y_best"] = y_best_np * _ones
            self.history_df["f_best"] = f_best_np * _ones
            self.history_df["sampler"] = sampler_name
        else:
            # Update history_df with new data
            X_batch = X[-self.batch_size:, :].cpu().numpy()
            Y_batch = Y[-self.batch_size:, :].cpu().numpy()
            F_batch = F[-self.batch_size:, :].cpu().numpy()
            
            df = pd.DataFrame(
                np.concatenate([X_batch, Y_batch, F_batch], axis=1),
                columns=[f"x{i}" for i in range(X.shape[1])] + ["Y", "F"]
            )

            # Compute the cumulative best `y_best` and `f_best`
            prev_y_best = self.history_df["y_best"].iloc[-1]
            prev_f_best = self.history_df["f_best"].iloc[-1]
            
            if self.is_maximize:
                new_y_best = max(prev_y_best, y_best_np.item())
                new_f_best = max(prev_f_best, f_best_np.item())
            else:
                new_y_best = min(prev_y_best, y_best_np.item())
                new_f_best = min(prev_f_best, f_best_np.item())

            _ones = np.ones_like(Y_batch)
            df["y_best"] = new_y_best * _ones
            df["f_best"] = new_f_best * _ones
            df["sampler"] = sampler_name

            self.history_df = pd.concat([self.history_df, df], ignore_index=True)

        if save_path:
            write_header = not os.path.exists(save_path)
            self.history_df.tail(self.batch_size).to_csv(
                save_path, mode='a', header=write_header, index=False
            )

    def report(self, max_rows: Optional[int] = None, save_path: Optional[str] = None):
        if self.history_df is None:
            logging.info("No data to report.")
            return

        if max_rows is not None:
            logging.info(tabulate(self.history_df.head(max_rows), headers='keys', tablefmt='psql'))
        else:
            logging.info(tabulate(self.history_df, headers='keys', tablefmt='psql'))

        # Plot Y and trueY as subplots with shared y-axis using plotly
        fig = make_subplots(rows=1, cols=2, shared_yaxes=True, subplot_titles=("Objective Values with Noise", "True Objective Values"))

        fig.add_trace(
            go.Scatter(
                x=self.history_df.index, 
                y=self.history_df["Y"],
                mode='markers', 
                name='Y (with noise)', 
                marker_color='blue'
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=self.history_df.index, 
                y=self.history_df["y_best"], 
                mode='lines', 
                name='Best Y (with noise)', 
                marker_color='red'
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=self.history_df.index, 
                y=self.history_df["F"], 
                mode='markers', 
                name='F (without noise)', 
                marker_color='blue'
            ),
            row=1, col=2
        )

        fig.add_trace(
            go.Scatter(
                x=self.history_df.index, 
                y=self.history_df["f_best"], 
                mode='lines', 
                name='Best F (without noise)', 
                marker_color='red'
            ),
            row=1, col=2
        )

        fig.update_layout(
            height=600, 
            width=1400, 
            title_text="Optimization Results",
            yaxis_title="Objective Value"
        )

        # Adjust axes titles for individual subplots
        fig.update_xaxes(title_text="Iteration", row=1, col=1)
        fig.update_xaxes(title_text="Iteration", row=1, col=2)

        if save_path:
            fig.write_image(save_path)
        else:
            fig.show()
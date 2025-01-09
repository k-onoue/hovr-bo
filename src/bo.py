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
        df = pd.read_csv(filepath)
        n_dim = len([col for col in df.columns if col.startswith('x')])
        data = df.to_numpy()
        
        self.X_all = torch.tensor(data[:, :n_dim], dtype=self.dtype, device=self.device)
        self.Y_all = torch.tensor(data[:, n_dim], dtype=self.dtype, device=self.device)
        self.F_all = torch.tensor(data[:, n_dim + 1], dtype=self.dtype, device=self.device)
        self.history_df = df
        
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

        for _ in range(self.n_iter):
            X_new = self.relative_sampler.sample(self.X_all, self.Y_all)
            Y_new, F_new = self._evaluate(X_new)
            
            self.X_all = torch.cat([self.X_all, X_new])
            self.Y_all = torch.cat([self.Y_all, Y_new])
            self.F_all = torch.cat([self.F_all, F_new])
            
            self._record("relative", save_path=save_path)

    def _evaluate(self, X: torch.Tensor):
        Y = self.objective_function(X)
        F = self.objective_function.evaluate_true(X)
        return Y, F

    def _record(self, sampler_name: str, save_path: Optional[str] = None):
        X = self.X_all.cpu().numpy()
        Y = self.Y_all.cpu().numpy()
        F = self.F_all.cpu().numpy()
        
        if self.is_maximize:
            y_best = torch.max(self.Y_all)
            f_best = torch.max(self.F_all)
        else:
            y_best = torch.min(self.Y_all)
            f_best = torch.min(self.F_all)
        
        data = []
        for i in range(len(X)):
            row = {f'x{j}': X[i,j] for j in range(X.shape[1])}
            row.update({
                'Y': Y[i],
                'F': F[i],
                'y_best': y_best.item(),
                'f_best': f_best.item(),
                'sampler': sampler_name
            })
            data.append(row)
            
        new_df = pd.DataFrame(data)
        
        if self.history_df is None:
            self.history_df = new_df
        else:
            self.history_df = pd.concat([self.history_df, new_df], ignore_index=True)
            
        if save_path:
            self.history_df.to_csv(save_path, index=False)

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


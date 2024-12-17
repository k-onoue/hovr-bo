from typing import Callable, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tabulate import tabulate
import torch

from .sampler_wrapper import IndependentSampler, RelativeSampler


class BayesianOptimization:
    def __init__(
        self,
        objective_function: Callable,
        sampler: Callable,
        n_initial_eval: int = 5,
        n_iter: int = 30,
        batch_size: int = 1,
        is_maximize: bool = False,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float64,
    ):
        self.objective_function = objective_function
        self.sampler = sampler
        self.device = device
        self.dtype = dtype

        self.n_iter = n_iter
        self.batch_size = batch_size
        self.is_maximize = is_maximize

        self.bounds = torch.tensor(self.objective_function._bounds, device=self.device, dtype=self.dtype).t()
    
        self.indenpendent_sampler = IndependentSampler(
            n_initial_eval=n_initial_eval,
            bounds=self.bounds,
            sample_method="sobol",
            dtype=self.dtype,
            device=self.device,
        )

        self.sampler = sampler

        self.history_df = None

    def _evaluate(self, X):
        Y = self.objective_function(X)
        F = self.objective_function.evaluate_true(X)

        if self.is_maximize:
            y_best = Y.max()
            f_best = F.max()
        else:
            y_best = Y.min()
            f_best = F.min()

        return Y, F, y_best, f_best

    def run(self):
        X = self.indenpendent_sampler.sample()
        Y, F, y_best, f_best = self._evaluate(X)

        self._record(X, Y, F, y_best, f_best, "independent")

        for iter in range(self.n_iter):
            relative_sampler = RelativeSampler(
                train_X=X,
                train_Y=Y if self.is_maximize else -Y,
                sampler=self.sampler,
                bounds=self.bounds,
                batch_size=self.batch_size,
                dtype=self.dtype,
                device=self.device,
            )

            candidates = relative_sampler.sample()
            X = torch.cat([X, candidates], dim=0)
            Y, F, y_best, f_best = self._evaluate(X)

            self._record(X, Y, F, y_best, f_best, "relative")

            print(f"Iteration {iter+1}/{self.n_iter} completed.")

        print("Optimization completed.")

    def _record(
            self, 
            X, 
            Y, 
            F, 
            y_best,
            f_best,
            sampler_name: str
        ):
        """
        Record the optimization history.

        Columns should include:
        - Objective variables (one column per dimension, e.g., x0, x1, ...)
        - Objective values with noise (Y)
        - Objective values without noise (true values, F)
        - Best objective value with noise so far (y_best)
        - Best objective value without noise so far (f_best)
        - Sampler name (e.g., independent or relative)
        """
        # Ensure Y and F are 2D tensors
        Y = Y.unsqueeze(-1) if Y.dim() == 1 else Y
        F = F.unsqueeze(-1) if F.dim() == 1 else F

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

            # Assign cumulative best to the batch
            df["y_best"] = new_y_best
            df["f_best"] = new_f_best
            df["sampler"] = sampler_name

            # Append the new batch to history_df
            self.history_df = pd.concat([self.history_df, df], axis=0).reset_index(drop=True)

    def report(self, max_rows: Optional[int] = None):
        if self.history_df is None:
            print("No data to report.")
            return

        if max_rows is not None:
            print(tabulate(self.history_df.head(max_rows), headers='keys', tablefmt='psql'))
        else:
            print(tabulate(self.history_df, headers='keys', tablefmt='psql'))

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

        fig.show()


from functools import wraps

import plotly.graph_objects as go
import torch

from .laplace_bnn import LaplaceBNN


def plot_fit(func):
    """
    Decorator to plot model fit on training data using Plotly.
    Works for 1D input data.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Extract input arguments safely
        train_X = kwargs.get("train_X")
        train_Y = kwargs.get("train_Y")
        bounds = kwargs.get("bounds")
        device = kwargs.get("device", torch.device("cpu"))
        dtype = kwargs.get("dtype", torch.float64)

        # Fall back to args if kwargs are missing
        if train_X is None and len(args) > 0:
            train_X = args[0]
        if train_Y is None and len(args) > 1:
            train_Y = args[1]
        if bounds is None and len(args) > 2:
            bounds = args[2]

        # Check if mandatory arguments are provided
        if train_X is None or train_Y is None or bounds is None:
            raise ValueError("train_X, train_Y, and bounds must be provided.")

        # Call the original function to fit and get the surrogate model
        candidates = func(*args, **kwargs)

        # Generate test points for plotting (dense grid in bounds)
        test_X = torch.linspace(bounds[0, 0], bounds[1, 0], steps=100).to(device, dtype)
        test_X = test_X.unsqueeze(-1)  # Shape: [100, 1]

        # LaplaceBNN surrogate model
        surrogate_model = LaplaceBNN(
            kwargs,
            input_dim=train_X.shape[-1],
            output_dim=train_Y.shape[-1],
            device=device,
            dtype=dtype,
        )
        surrogate_model.fit(train_X, train_Y)
        posterior = surrogate_model.posterior(test_X)
        pred_Y = posterior.mean.squeeze(1)  # Use posterior mean for predictions
        pred_std = torch.sqrt(posterior.variance).squeeze(1)  # Standard deviation for credible intervals

        # Compute upper and lower credible intervals
        lower_bound = (pred_Y - 2 * pred_std).detach().cpu().numpy()
        upper_bound = (pred_Y + 2 * pred_std).detach().cpu().numpy()

        # Convert tensors to numpy for plotting
        train_X_np = train_X.cpu().numpy().flatten()
        train_Y_np = train_Y.cpu().numpy().flatten()
        test_X_np = test_X.cpu().numpy().flatten()
        pred_Y_np = pred_Y.detach().cpu().numpy().flatten()

        # Plotly visualization
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=train_X_np, y=train_Y_np,
                                 mode='markers', name='Training Data'))
        fig.add_trace(go.Scatter(x=test_X_np, y=pred_Y_np,
                                 mode='lines', name='Model Prediction'))
        fig.add_trace(go.Scatter(x=test_X_np, y=upper_bound,
                                 mode='lines', name='Upper Bound', line=dict(dash='dot')))
        fig.add_trace(go.Scatter(x=test_X_np, y=lower_bound,
                                 mode='lines', name='Lower Bound', line=dict(dash='dot')))
        fig.update_layout(
            title=f"Model Fit with Credible Intervals: {func.__name__}",
            xaxis_title="Input X",
            yaxis_title="Output Y",
        )
        fig.show()

        return candidates

    return wrapper

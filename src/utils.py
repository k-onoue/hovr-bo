import numpy as np
import plotly.graph_objects as go
import torch


# Create toy y = x^3 + noise data
def generate_toy_data():
    torch.manual_seed(0)
    x_train = torch.linspace(-4, 4, 1000).unsqueeze(-1)
    sigma = torch.normal(torch.zeros_like(x_train), 3 * torch.ones_like(x_train))
    y_train = x_train**3 + sigma

    x_test = torch.linspace(-7, 7, 1000).unsqueeze(-1)
    y_test = x_test**3

    return x_train, y_train, x_test, y_test


def generate_data(
    func,
    x_min_train,
    x_max_train,
    num_train=100,
    num_test=1000,
    sigma_scale=0.2,
    test_additional=0.4,
    outlier_ratio=0.01,
    outlier_value_range=(5.0, 5.1),
):
    # Training data
    x_train = torch.linspace(x_min_train, x_max_train, num_train).unsqueeze(-1)
    sigma = torch.normal(
        torch.zeros_like(x_train), sigma_scale * torch.ones_like(x_train)
    )
    y_train = func(x_train) + sigma

    # Adding outliers to the training data
    n_outliers = int(num_train * outlier_ratio)
    outlier_indices = np.random.choice(num_train, n_outliers, replace=False)

    # Make sure the assigned outlier values match the shape of y_train
    y_train[outlier_indices] = outlier_value_range[0] + torch.rand(
        n_outliers
    ).unsqueeze(-1) * (outlier_value_range[1] - outlier_value_range[0])

    # Automatically set the test range based on the ratio (test_additional)
    test_range = abs(x_max_train - x_min_train) * test_additional
    x_min_test = x_min_train - test_range
    x_max_test = x_max_train + test_range

    # Test data
    x_test = torch.linspace(x_min_test, x_max_test, num_test).unsqueeze(-1)
    y_test = func(x_test)

    return x_train, y_train, x_test, y_test, outlier_indices


def plot_data(x_train, y_train, x_test, y_test, outlier_indices):
    # Convert tensors to numpy arrays for plotting
    x_train_np = x_train.numpy().flatten()
    y_train_np = y_train.numpy().flatten()
    x_test_np = x_test.numpy().flatten()
    y_test_np = y_test.numpy().flatten()

    # Create a scatter plot for training data
    train_scatter = go.Scatter(
        x=x_train_np,
        y=y_train_np,
        mode="markers",
        name="Training Data",
        marker=dict(color="blue"),
    )

    # Highlight the outliers in the training data
    outlier_scatter = go.Scatter(
        x=x_train_np[outlier_indices],
        y=y_train_np[outlier_indices],
        mode="markers",
        name="Outliers",
        marker=dict(color="red", size=10, symbol="x"),
    )

    # Create a scatter plot for test data
    test_scatter = go.Scatter(
        x=x_test_np,
        y=y_test_np,
        mode="lines",
        name="Test Data (True Function)",
        line=dict(color="green"),
    )

    # Create the figure and add traces
    fig = go.Figure()
    fig.add_trace(train_scatter)
    fig.add_trace(outlier_scatter)
    fig.add_trace(test_scatter)

    # Customize the layout
    fig.update_layout(
        title="Training and Test Data with Outliers",
        xaxis_title="X",
        yaxis_title="Y",
        legend=dict(x=0, y=1.1),
        height=600,
        width=900,
    )

    # Show the plot
    fig.show()


# Example usage
x_train, y_train, x_test, y_test, outlier_indices = generate_data(
    lambda x: torch.sin(x),
    0,
    2 * torch.pi,
    num_train=300,
    test_additional=0,
    num_test=1000,
    outlier_ratio=0.0,
    outlier_value_range=(5.0, 5.1),
)
plot_data(x_train, y_train, x_test, y_test, outlier_indices)

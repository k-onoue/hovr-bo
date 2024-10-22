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

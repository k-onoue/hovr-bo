import torch
from torch import nn
import plotly.graph_objects as go
from edl_pytorch import NormalInvGamma
from edl_pytorch.loss import nig_nll, nig_reg


torch.manual_seed(0)

# Define the true function
true_func = lambda x: torch.sin(2 * x) * x 

def generate_data(func, x_min_train, x_max_train, num_train=1000, num_test=1000, sigma_scale=3, test_additional=0.4):
    # Training data
    x_train = torch.linspace(x_min_train, x_max_train, num_train).unsqueeze(-1)
    sigma = torch.normal(torch.zeros_like(x_train), sigma_scale * torch.ones_like(x_train))
    y_train = func(x_train) + sigma

    # Automatically set the test range based on the ratio (test_add)
    test_range = abs(x_max_train - x_min_train) * test_additional
    x_min_test = x_min_train - test_range
    x_max_test = x_max_train + test_range

    # Test data
    x_test = torch.linspace(x_min_test, x_max_test, num_test).unsqueeze(-1)
    y_test = func(x_test)
    
    return x_train, y_train, x_test, y_test


# Generate train and test data
x_train, y_train, x_test, y_test = generate_data(
    true_func, 
    x_min_train=4, 
    x_max_train=-4,
    num_train=100,
    num_test=1000,
    sigma_scale=1,
    test_additional=0,
)


class EvidentialMLP(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_units=[64, 64, 64],
        activations=["tanh", "tanh", "tanh"],
        nig_coeff=1e-2,
        epoch_num=500,
        lr=5e-4
    ):
        super(EvidentialMLP, self).__init__()
        layers = []
        in_dim = input_dim

        # Build layers based on the given hidden_units and activations
        for hidden_dim, activation in zip(hidden_units, activations):
            layers.append(nn.Linear(in_dim, hidden_dim))
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            elif activation == "sigmoid":
                layers.append(nn.Sigmoid())
            in_dim = hidden_dim

        layers.append(NormalInvGamma(hidden_units[-1], output_dim))
        self.model = nn.Sequential(*layers)

        # Set coefficients for loss components
        self.nig_coeff = nig_coeff

        # Optimizer parameters
        self.epoch_num = epoch_num
        self.lr = lr

    def forward_dist_params(self, x):
        return self.model(x)

    def forward(self, x):
        with torch.no_grad():
            dist_params = self.forward_dist_params(x)
            mu, v, alpha, beta = (d.squeeze() for d in dist_params)
            std = torch.sqrt(beta / (v * (alpha - 1)))
        return mu, std
    
    def compute_loss(self, x, y):
        dist_params = self.forward_dist_params(x)
        return nig_nll(*dist_params, y) + self.nig_coeff * nig_reg(*dist_params, y)
    
    def fit(self, x_train, y_train):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
    
        for epoch in range(self.epoch_num):
            # Full batch: x_train and y_train used as a whole
            loss = self.compute_loss(x_train, y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch % 10 == 0 or epoch == self.epoch_num - 1:
                print(f'Epoch: {epoch}, Loss: {loss.item()}')


# Create Evidential MLP model
model = EvidentialMLP(
    input_dim=1, 
    output_dim=1,
    hidden_units=[64, 64, 64],
    activations=["tanh", "tanh", "tanh"],
    nig_coeff=1e-2,
    epoch_num=500,
    lr=5e-4
)

# Train the model
model.fit(x_train, y_train)

# Make predictions on test data
mean, std = model(x_test)

# Prepare data for visualization
mean, std = mean.squeeze().numpy(), std.squeeze().numpy()
x_test, y_test = x_test.squeeze().numpy(), y_test.squeeze().numpy()

# Plot using Plotly
fig = go.Figure()

# Add train data points
fig.add_trace(go.Scatter(x=x_train.squeeze().numpy(), y=y_train.squeeze().numpy(), mode='markers', 
                         marker=dict(size=3, color='blue'), name='Train'))

# Add true function line
fig.add_trace(go.Scatter(x=x_test, y=y_test, mode='lines', 
                         line=dict(color='black', width=2), name='True'))

# Add predicted mean line
fig.add_trace(go.Scatter(x=x_test, y=mean, mode='lines', 
                         line=dict(color='blue', dash='dash'), name='Pred'))

# Add uncertainty (2 std devs only)
fig.add_trace(go.Scatter(
    x=x_test, 
    y=(mean - 2 * std), 
    fill=None, mode='lines', line_color='rgba(0,0,255,0.1)',  # Lighter color
    showlegend=True, name='Unc. (2σ)'
))
fig.add_trace(go.Scatter(
    x=x_test, 
    y=(mean + 2 * std), 
    fill='tonexty', mode='lines', line_color='rgba(0,0,255,0.1)',
    showlegend=False
))

# Update layout with white background
fig.update_layout(
    title="Evidential Regression",
    xaxis_title="x",
    yaxis_title="y",
    legend=dict(x=0, y=1),
    plot_bgcolor='white',  # White background
    paper_bgcolor='white'  # White background
)

fig.show()
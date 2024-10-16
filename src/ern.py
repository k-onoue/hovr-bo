import numpy as np
import torch
from edl_pytorch.layer import NormalInvGamma
from edl_pytorch.loss import nig_nll, nig_reg
from torch import nn


# Uncertainty Regularization to Bypass HUA
# From https://arxiv.org/abs/2401.01484
# Based on the equation from the paper (equation 17)
def u_reg(gamma, _v, alpha, _beta, y):
    error = (y - gamma).abs()  # |y - gamma|
    uncertainty = torch.log(torch.exp(alpha - 1) - 1)  # log(exp(alpha - 1) - 1)
    reg = -error * uncertainty  # Combine the terms as shown in the formula
    return reg.mean()  # Return the mean value of the regularization term


# Non-saturating Uncertainty Regularizer
# From https://ojs.aaai.org/index.php/AAAI/article/view/30172:
# Based on the equation from the paper (equation 20)
def nsu_reg(gamma, v, alpha, beta, y):
    # (y - gamma)^2 * (v * (alpha - 1)) / (beta * (v + 1))
    error = (y - gamma).pow(2)
    uncertainty = v * (alpha - 1) / (beta * (v + 1))
    reg = error * uncertainty
    return reg.mean()





# HOVR regularization adapted for the EvidentialMLP model
def hovr_reg(model, x, k=3, q=2, num_points=10):
    # Generate random points within the input range
    x_min, x_max = x.min(0)[0], x.max(0)[0]
    random_points = torch.tensor(
        np.random.uniform(x_min.numpy(), x_max.numpy(), (num_points, x.shape[1])),
        dtype=torch.float32,
        requires_grad=True,
    )

    # Compute the model's output for the random points (mu only)
    mu, _, _, _ = model(random_points)  # Model returns a tuple (mu, v, alpha, beta)

    # Compute gradients of the output with respect to the random points
    grads = torch.autograd.grad(
        mu, random_points, torch.ones_like(mu), create_graph=True
    )[0]

    # Calculate HOVR term based on the gradients
    hovr_term = 0.0
    for i in range(x.shape[1]):  # Calculate k-th order derivative for each dimension
        grad_i = grads[:, i]
        temp_grad = grad_i
        for _ in range(k - 1):
            temp_grad = torch.autograd.grad(
                temp_grad, random_points, torch.ones_like(temp_grad), create_graph=True
            )[0][:, i]
        hovr_term += torch.sum(torch.abs(temp_grad) ** q)

    return hovr_term / x.shape[1]  # Normalize by the input dimension


# Modular EvidentialMLP model
class EvidentialMLP(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_units,
        activations,
        nig_coeff=1.0,
        nsu_coeff=0.0,
        hovr_coeff=0.0,
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
            # Additional activation functions can be added here
            in_dim = hidden_dim

        # Final output layer (NormalInvGamma)
        layers.append(NormalInvGamma(hidden_units[-1], output_dim))

        # Combine the layers into a Sequential model
        self.model = nn.Sequential(*layers)

        # Set coefficients for NIG, NSU, and HOVR regularization
        self.nig_coeff = nig_coeff
        self.nsu_coeff = nsu_coeff
        self.hovr_coeff = hovr_coeff

    def forward(self, x):
        return self.model(x)

    # Method to compute the mean and standard deviation from NIG output
    def predict(self, x, pred=None):
        if not pred:
            pred = self.forward(x)
        mu, v, alpha, beta = (d.squeeze() for d in pred)
        std = torch.sqrt(beta / (v * (alpha - 1)))
        return mu, std

    # Loss function method for NIG, NSU, and HOVR regularization
    def compute_loss(self, dist_params, y, x=None):
        """
        Compute the loss function, including NIG loss and optional NSU and HOVR regularization.
        :param dist_params: Tuple (mu, v, alpha, beta) from NormalInvGamma layer
        :param y: Target values
        :param x: Input data, required for HOVR regularization
        :return: Computed loss
        """
        # NIG negative log-likelihood and regularization
        loss = nig_nll(*dist_params, y) + self.nig_coeff * nig_reg(*dist_params, y)

        # NSU regularization if the coefficient is non-zero
        if self.nsu_coeff > 0.0:
            loss += self.nsu_coeff * nsu_reg(*dist_params, y)

        # HOVR regularization if the coefficient is non-zero
        if self.hovr_coeff > 0.0:
            assert x is not None, "Input `x` is required for HOVR regularization."
            loss += self.hovr_coeff * hovr_reg(self, x)

        return loss



#############################################################################
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np

# Create toy y = x^3 + noise data
def generate_toy_data():
    torch.manual_seed(0)
    x_train = torch.linspace(-4, 4, 1000).unsqueeze(-1)
    sigma = torch.normal(torch.zeros_like(x_train), 3 * torch.ones_like(x_train))
    y_train = x_train**3 + sigma

    x_test = torch.linspace(-7, 7, 1000).unsqueeze(-1)
    y_test = x_test**3

    return x_train, y_train, x_test, y_test

# Train the model on the toy dataset
def train_model(model, x_train, y_train, epochs=500, batch_size=100, lr=5e-4):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        for x, y in DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True):
            pred = model(x)
            loss = model.compute_loss(pred, y, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

# Plot the prediction results
def plot_results(model, x_train, y_train, x_test, y_test, save_path="cubic_regression.png"):
    with torch.no_grad():
        pred = model(x_test)
        mu, std = model.predict(x_test, pred)

    plt.figure(figsize=(6, 4), dpi=200)
    plt.scatter(x_train, y_train, s=1.0, c="tab:blue", label="Train")
    plt.plot(x_test, y_test, c="k", label="True")
    plt.plot(x_test.squeeze(), mu, c="tab:blue", ls="--", label="Pred")
    plt.fill_between(
        x_test.squeeze(), (mu - 2 * std), (mu + 2 * std),
        alpha=0.2, facecolor="tab:blue", label="Unc."
    )
    plt.gca().set_ylim(-150, 150)
    plt.gca().set_xlim(-7, 7)
    plt.legend(loc="upper left")
    plt.savefig(save_path)
    plt.show()

if __name__ == "__main__":
    # Generate toy dataset
    x_train, y_train, x_test, y_test = generate_toy_data()

    # Initialize model with customized settings
    model_config = {
        "input_dim": 1,
        "output_dim": 1,
        "hidden_units": [128, 128, 128],
        "activations": ["tanh", "relu", "relu"],
        "nig_coeff": 1e-2,
        # "nsu_coeff": 1e-3,
        "nsu_coeff": 0,
        "hovr_coeff": 1e-4,
        # "hovr_coeff": 0,
    }
    model = EvidentialMLP(**model_config)

    # Train the model
    train_model(model, x_train, y_train)

    # Plot the results
    plot_results(model, x_train, y_train, x_test, y_test)
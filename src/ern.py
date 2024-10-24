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


# Lipschitz Modified MSE Regularization
# Implemented based on the equation in the paper (equation 6)
def lipschitz_mse_reg(gamma, v, alpha, beta, y):
    # Calculate the squared error (y - gamma)^2
    error = (y - gamma).pow(2)
    
    # Calculate U_alpha and U_nu based on the given formula
    U_nu = beta * (1 + v) / (alpha * v)
    U_alpha = 2 * beta * (1 + v) / v * (torch.exp(torch.digamma(alpha + 0.5) - torch.digamma(alpha)) - 1)

    # Minimum U_nu and U_alpha value to control the Lipschitz constant
    U_nu_alpha = torch.min(U_nu, U_alpha)

    # Lipschitz Modified MSE: switch between normal MSE and Lipschitz adjusted term
    modified_mse = torch.where(
        error < U_nu_alpha, 
        error, 
        2 * torch.sqrt(U_nu_alpha) * torch.abs(y - gamma) - U_nu_alpha
    )

    return modified_mse.mean()


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
        lipschitz_coeff=0.0,  # Added for Lipschitz regularization
        l2_coeff=0.0
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

        # Final output layer (NormalInvGamma)
        self.normal_inv_gamma = NormalInvGamma(hidden_units[-1], output_dim)

        # Combine the layers into a Sequential model for MLP
        self.mlp = nn.Sequential(*layers)

        # Set coefficients for NIG, NSU, HOVR, Lipschitz MSE, and L2 regularization
        self.nig_coeff = nig_coeff
        self.nsu_coeff = nsu_coeff
        self.hovr_coeff = hovr_coeff
        self.lipschitz_coeff = lipschitz_coeff  # Added for Lipschitz regularization
        self.l2_coeff = l2_coeff

    def forward(self, x):
        x = self.mlp(x)
        return self.normal_inv_gamma(x)

    # Method to compute the mean and standard deviation from NIG output
    def predict(self, x, pred=None):
        if not pred:
            pred = self.forward(x)
        mu, v, alpha, beta = (d.squeeze() for d in pred)
        std = torch.sqrt(beta / (v * (alpha - 1)))
        return mu, std

    # Loss function method for NIG, NSU, HOVR, Lipschitz MSE, and L2 regularization
    def compute_loss(self, dist_params, y, x=None):
        """
        Compute the loss function, including NIG loss and optional NSU, HOVR, Lipschitz MSE, and L2 regularization.
        :param dist_params: Tuple (mu, v, alpha, beta) from NormalInvGamma layer
        :param y: Target values
        :param x: Input data, required for HOVR regularization
        :return: Computed loss
        """
        # NIG negative log-likelihood and regularization
        loss = nig_nll(*dist_params, y) 
        
        if self.nig_coeff > 0.0:
            loss += self.nig_coeff * nig_reg(*dist_params, y)

        # NSU regularization if the coefficient is non-zero
        if self.nsu_coeff > 0.0:
            loss += self.nsu_coeff * nsu_reg(*dist_params, y)

        # HOVR regularization if the coefficient is non-zero
        if self.hovr_coeff > 0.0:
            assert x is not None, "Input `x` is required for HOVR regularization."
            loss += self.hovr_coeff * hovr_reg(self, x)

        # Lipschitz MSE regularization if the coefficient is non-zero
        if self.lipschitz_coeff > 0.0:
            loss += self.lipschitz_coeff * lipschitz_mse_reg(*dist_params, y)

        # L2 Regularization (Weight Decay) only on MLP layers (excluding NormalInvGamma)
        if self.l2_coeff > 0.0:
            l2_norm = sum(p.pow(2.0).sum() for p in self.mlp.parameters())  # L2 norm only for MLP parameters
            loss += self.l2_coeff * l2_norm

        return loss


# Training function for EvidentialMLP model
def train_ern(model, X_tensor, y_tensor, num_epochs=500, lr=0.01):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(num_epochs):
        
        model.train()
        optimizer.zero_grad()
        
        # Full batch passed to model
        preds = model(X_tensor)
        
        # Compute the loss
        loss = model.compute_loss(preds, y_tensor, X_tensor)
        
        # Backpropagation and optimization
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch + 1}/{num_epochs}: Loss={loss.item():.4f}")

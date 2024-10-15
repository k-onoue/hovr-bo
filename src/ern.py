import numpy as np
import torch
from edl_pytorch.layer import NormalInvGamma
from edl_pytorch.loss import nig_nll, nig_reg
from torch import nn


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
def hovr_reg(model, x, k=2, q=2, num_points=10):
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
    def compute_mean_and_std(self, pred):
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


# if __name__ == "__main__":

#     # モデルを初期化する関数
#     def create_model(config):
#         input_dim = config["input_dim"]
#         output_dim = config["output_dim"]
#         hidden_layers = config["hidden_layers"]
#         activations = config["activations"]

#         return EvidentialMLP(input_dim, output_dim, hidden_layers, activations)

#     model_dict = {
#         "ern1": {
#             "model": "vanilla ern",
#             "input_dim": 1,
#             "output_dim": 1,
#             "hidden_layers": [64, 64],
#             "activations": ["relu", "relu"],
#         },
#         "ern2": {
#             "model": "deep ern",
#             "input_dim": 1,
#             "output_dim": 1,
#             "hidden_layers": [128, 128, 64],
#             "activations": ["relu", "tanh", "relu"],
#         },
#     }

#     # 辞書からモデルを作成
#     model1 = create_model(model_dict["ern1"])
#     model2 = create_model(model_dict["ern2"])

#     print(model1)
#     print()
#     print(model2)

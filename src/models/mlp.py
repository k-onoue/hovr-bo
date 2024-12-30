import torch
import torch.nn as nn
from typing import Union


class MLP(nn.Sequential):
    def __init__(
        self,
        dimensions: list[int],
        activation: str,
        input_dim: int = 1,
        output_dim: int = 1,
        dtype: torch.dtype = torch.float64,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        """
        A Multi-Layer Perceptron (MLP) model.

        Args:
            dimensions (list[int]): List of hidden layer dimensions.
            activation (str): Activation function ('tanh', 'relu', 'elu').
            input_dim (int): Dimension of the input.
            output_dim (int): Dimension of the output.
            dtype (torch.dtype): Data type for the model's parameters.
            device (Union[str, torch.device]): Device for the model.
        """
        super(MLP, self).__init__()

        # Combine input, hidden, and output dimensions
        self.dimensions: list[int] = [input_dim, *dimensions, output_dim]

        # Loop to create layers
        for i in range(len(self.dimensions) - 1):
            # Add linear layer
            self.add_module(
                f"linear{i}",
                nn.Linear(
                    self.dimensions[i],
                    self.dimensions[i + 1],
                    dtype=dtype,
                    device=device,
                ),
            )

            # Add activation function for hidden layers
            if i < len(self.dimensions) - 2:
                if activation == "tanh":
                    self.add_module(f"activation{i}", nn.Tanh())
                elif activation == "relu":
                    self.add_module(f"activation{i}", nn.ReLU())
                elif activation == "elu":
                    self.add_module(f"activation{i}", nn.ELU())
                else:
                    raise NotImplementedError(
                        f"Activation type '{activation}' is not supported."
                    )

            # # Add activation function for hidden layers
            # if i < len(self.dimensions) - 2:
            #     if activation == "tanh":
            #         self.add_module(f"tanh{i}", nn.Tanh())
            #     elif activation == "relu":
            #         self.add_module(f"relu{i}", nn.ReLU())
            #     elif activation == "elu":
            #         self.add_module(f"elu{i}", nn.ELU())
            #     else:
            #         raise NotImplementedError(
            #             f"Activation type '{activation}' is not supported."
            #         )
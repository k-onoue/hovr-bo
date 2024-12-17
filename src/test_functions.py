import torch
from botorch.test_functions.base import BaseTestProblem 
from torch import Tensor
from typing import Optional


class OutlierTestProblem(BaseTestProblem):
    r"""
    Extension of BaseTestProblem to add functionality for injecting outliers into the function values.
    Outliers are added based on a specified probability and scale.
    """
    def __init__(
        self,
        noise_std: Optional[float] = None,
        negate: bool = False,
        outlier_prob: Optional[float] = None,
        outlier_scale: float = 10.0,
    ):
        r"""
        Args:
            noise_std: Standard deviation of the observation noise.
            negate: If True, negate the function output.
            outlier_prob: Probability of an outlier occurring for each function evaluation. 
                          If None, no outliers will be added.
            outlier_scale: Magnitude scale of the outliers.
        """
        super().__init__(noise_std=noise_std, negate=negate)
        self.outlier_prob = outlier_prob  # Probability threshold for outlier occurrence
        self.outlier_scale = outlier_scale  # Scale factor for outlier magnitude

    def forward(self, X: Tensor, noise: bool = True) -> Tensor:
        r"""
        Evaluate the function, optionally adding observation noise and injecting outliers.

        Args:
            X: A `(batch_shape) x d`-dim tensor of input points.
            noise: If True, add observation noise to the function output.

        Returns:
            A `batch_shape`-dim tensor of function evaluations with optional noise and outliers.
        """
        f = self.evaluate_true(X=X)
        
        if noise and self.noise_std is not None:
            _noise = torch.tensor(self.noise_std, device=X.device, dtype=X.dtype)
            f += _noise * torch.randn_like(f)

        if self.outlier_prob is not None:
            outlier_mask = torch.rand_like(f) < self.outlier_prob
            _outlier = torch.tensor(self.outlier_scale, device=X.device, dtype=X.dtype)
            outliers = _outlier * torch.randn_like(f)
            f = torch.where(outlier_mask, f + outliers, f)

        if self.negate:
            f = -f
        
        return f  


class SyntheticSine(OutlierTestProblem):
    r"""
    Synthetic test function defined as:
        f(x) = - (x - 1)^2 * sin(3x + 5/x + 1),  where x ∈ [5, 10].
    """

    def __init__(
        self, 
        noise_std: Optional[float] = None, 
        negate: bool = False,
        outlier_prob: Optional[float] = None,
        outlier_scale: float = 10.0,
    ):
        r"""
        Args:
            noise_std: Standard deviation of the observation noise.
            negate: If True, negate the function.
            outlier_prob: Probability of an outlier occurring for each function evaluation. 
                          If None, no outliers will be added.
            outlier_scale: Magnitude scale of the outliers.
        """
        self.dim = 1  # 1D function
        self._bounds = [(5.0, 10.0)]  # Domain bounds
        super().__init__(
            noise_std=noise_std, 
            negate=negate, 
            outlier_prob=outlier_prob, 
            outlier_scale=outlier_scale
        )
    
    def evaluate_true(self, X: Tensor) -> Tensor:
        r"""
        Compute the true function value.

        Args:
            X: A (batch_shape) x 1 tensor of input points.

        Returns:
            A tensor of function evaluations.
        """
        # Ensure X is within bounds
        X = X.clamp(min=self._bounds[0][0], max=self._bounds[0][1])
        
        # Compute the function: - (x - 1)^2 * sin(3x + 5/x + 1)
        term1 = torch.pow(X - 1, 2)  # (x - 1)^2
        term2 = torch.sin(3 * X + 5 / X + 1)  # sin(3x + 5/x + 1)
        result = - term1 * term2
        
        return result.squeeze(-1)


class BraninFoo(OutlierTestProblem):
    r"""
    Branin function with outliers:
        f(x_1, x_2) = (x_2 - 5.1/(4*pi^2) * x_1^2 + 5/pi * x_1 - 6)^2 
                      + 10 * (1 - 1/(8*pi)) * cos(x_1) + 10,
        where x_1 ∈ [0, 15], x_2 ∈ [-5, 15].
    """

    def __init__(
        self, 
        noise_std: Optional[float] = None, 
        negate: bool = False,
        outlier_prob: Optional[float] = None,
        outlier_scale: float = 10.0,
    ):
        r"""
        Args:
            noise_std: Standard deviation of the observation noise.
            negate: If True, negate the function.
            outlier_prob: Probability of an outlier occurring for each function evaluation. 
                          If None, no outliers will be added.
            outlier_scale: Magnitude scale of the outliers.
        """
        self.dim = 2  # 2D function
        self._bounds = [(0.0, 15.0), (-5.0, 15.0)]  # Domain bounds
        super().__init__(
            noise_std=noise_std, 
            negate=negate, 
            outlier_prob=outlier_prob, 
            outlier_scale=outlier_scale
        )
    
    def evaluate_true(self, X: Tensor) -> Tensor:
        r"""
        Compute the true function value.

        Args:
            X: A (batch_shape) x 2 tensor of input points.

        Returns:
            A tensor of function evaluations.
        """
        x1 = X[..., 0]
        x2 = X[..., 1]
        term1 = x2 - (5.1 / (4 * torch.pi**2)) * x1**2 + (5 / torch.pi) * x1 - 6
        term2 = 10 * (1 - 1 / (8 * torch.pi)) * torch.cos(x1)
        result = term1**2 + term2 + 10
        
        return result


# if __name__ == "__main__":
#     from plotly import graph_objects as go

#     # インスタンス化
#     noise_std = 2  # ノイズ標準偏差
#     outlier_prob = 0.05  # 外れ値発生確率
#     outlier_scale = 30  # 外れ値スケール
#     branin_function = BraninFoo(
#         noise_std=noise_std, 
#         outlier_prob=outlier_prob, 
#         outlier_scale=outlier_scale
#     )

#     # 入力点の生成
#     x1 = torch.linspace(0, 15, 100)
#     x2 = torch.linspace(-5, 15, 100)
#     X1, X2 = torch.meshgrid(x1, x2)
#     X = torch.stack([X1, X2], dim=-1).reshape(-1, 2)

#     # 真の関数値の計算
#     true_values = branin_function.evaluate_true(X).reshape(100, 100)

#     # ノイズ付き関数値の計算
#     noisy_values = branin_function(X, noise=True).reshape(100, 100)

#     # Plotlyによるプロット
#     fig = go.Figure()

#     # 真の関数のプロット
#     fig.add_trace(go.Surface(
#         z=true_values.numpy(),
#         x=X1.numpy(),
#         y=X2.numpy(),
#         colorscale="Viridis",
#         name="True Function"
#     ))

#     # ノイズを含んだ関数のプロット
#     fig.add_trace(go.Surface(
#         z=noisy_values.numpy(),
#         x=X1.numpy(),
#         y=X2.numpy(),
#         colorscale="Viridis",
#         opacity=0.7,
#         name="Noisy Observations"
#     ))

#     # レイアウト設定
#     fig.update_layout(
#         title="Branin Function with Noise",
#         scene=dict(
#             xaxis_title="x1",
#             yaxis_title="x2",
#             zaxis_title="f(x1, x2)"
#         ),
#         template="plotly",
#         height=700, width=900
#     )

#     fig.show()



# if __name__ == "__main__":
#     from plotly import graph_objects as go

#     # インスタンス化
#     noise_std = 2  # ノイズ標準偏差
#     outlier_prob = 0.05  # 外れ値発生確率
#     outlier_scale = 30  # 外れ値スケール
#     sine_function = SyntheticSine(
#         noise_std=noise_std, 
#         outlier_prob=outlier_prob, 
#         outlier_scale=outlier_scale
#     )

#     # 入力点の生成
#     X = torch.linspace(5, 10, 200).reshape(-1, 1)  # [5, 10] 区間で200点

#     # 真の関数値の計算
#     true_values = sine_function.evaluate_true(X)

#     # ノイズ付き関数値の計算
#     noisy_values = sine_function(X, noise=True)

#     # Plotlyによるプロット
#     fig = go.Figure()

#     # 真の関数のプロット
#     fig.add_trace(go.Scatter(
#         x=X.squeeze().numpy(),
#         y=true_values.numpy(),
#         mode="lines",
#         name="True Function",
#         line=dict(width=2)
#     ))

#     # ノイズを含んだ関数のプロット
#     fig.add_trace(go.Scatter(
#         x=X.squeeze().numpy(),
#         y=noisy_values.numpy(),
#         mode="markers",
#         name="Noisy Observations",
#         marker=dict(size=5, opacity=0.7)
#     ))

#     # レイアウト設定
#     fig.update_layout(
#         title="Synthetic Sine Function with Noise",
#         xaxis_title="x",
#         yaxis_title="f(x)",
#         template="plotly",
#         legend=dict(x=0, y=1),
#         height=500, width=800
#     )

#     fig.show()

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
        outlier_scale: float = 5.0,
        outlier_std: float = 1.0,
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
        self.outlier_std = outlier_std  # Standard deviation of the outlier noise

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

        # Inject outliers if outlier_prob is specified
        if self.outlier_prob is not None:
            outlier_mask = torch.rand_like(f) < self.outlier_prob  # Mask for outlier locations
            
            # Flip sign of outlier_scale with probability 1/2
            sign_flip = torch.where(torch.rand_like(f) > 0.5, 1.0, -1.0)
            _scale = torch.tensor(self.outlier_scale, device=X.device, dtype=X.dtype)
            _std = torch.tensor(self.outlier_std, device=X.device, dtype=X.dtype)
            base_outlier = sign_flip * _scale + _std * torch.randn_like(f)
            
            # Add outliers to the original function values
            f = torch.where(outlier_mask, f + base_outlier, f)

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
        outlier_scale: float = 5.0,
        outlier_std: float = 1.0,
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
        self._bounds = [(5.0, 10.0)] # Domain bounds: [(x_i min, x_i max),]
        super().__init__(
            noise_std=noise_std, 
            negate=negate, 
            outlier_prob=outlier_prob, 
            outlier_scale=outlier_scale,
            outlier_std=outlier_std
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
        
        # return result.squeeze(-1)
        return result


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
        outlier_scale: float = 5.0,
        outlier_std: float = 1.0,
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
        self._bounds = [(0.0, 15.0), (-5.0, 15.0)] # Domain bounds: [(x_i min, x_i max),]
        super().__init__(
            noise_std=noise_std, 
            negate=negate, 
            outlier_prob=outlier_prob, 
            outlier_scale=outlier_scale,
            outlier_std=outlier_std
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
        
        # return result
        return result.unsqueeze(-1)
    

class Ackley2d(OutlierTestProblem):
    r"""
    Ackley function in 2D with outliers:
        f(x_1, x_2) = -20 * exp(-0.2 * sqrt(0.5 * (x_1^2 + x_2^2))) 
                      - exp(0.5 * (cos(2*pi*x_1) + cos(2*pi*x_2))) + 20 + e,
        where x_1, x_2 ∈ [-5, 5].
    """

    def __init__(
        self, 
        noise_std: Optional[float] = None, 
        negate: bool = False,
        outlier_prob: Optional[float] = None,
        outlier_scale: float = 5.0,
        outlier_std: float = 1.0,
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
        self._bounds = [(-5.0, 5.0), (-5.0, 5.0)] # Domain bounds: [(x_i min, x_i max),]
        super().__init__(
            noise_std=noise_std, 
            negate=negate, 
            outlier_prob=outlier_prob, 
            outlier_scale=outlier_scale,
            outlier_std=outlier_std
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
        term1 = -20 * torch.exp(-0.2 * torch.sqrt(0.5 * (x1**2 + x2**2)))
        term2 = -torch.exp(0.5 * (torch.cos(2 * torch.pi * x1) + torch.cos(2 * torch.pi * x2)))
        result = term1 + term2 + 20 + torch.exp(torch.tensor(1.0, device=X.device, dtype=X.dtype))
        
        # return result
        return result.unsqueeze(-1)
    

class Ackley5d(OutlierTestProblem):
    r"""
    Ackley function in 5D with outliers:
        f(x_1, x_2, x_3, x_4, x_5) = -20 * exp(-0.2 * sqrt(0.5 * (x_1^2 + x_2^2 + x_3^2 + x_4^2 + x_5^2))) 
                                      - exp(0.5 * (cos(2*pi*x_1) + cos(2*pi*x_2) + cos(2*pi*x_3) + cos(2*pi*x_4) + cos(2*pi*x_5))) + 20 + e,
        where x_1, x_2, x_3, x_4, x_5 ∈ [-5, 5].
    """

    def __init__(
        self, 
        noise_std: Optional[float] = None, 
        negate: bool = False,
        outlier_prob: Optional[float] = None,
        outlier_scale: float = 5.0,
        outlier_std: float = 1.0,
    ):
        r"""
        Args:
            noise_std: Standard deviation of the observation noise.
            negate: If True, negate the function.
            outlier_prob: Probability of an outlier occurring for each function evaluation. 
                          If None, no outliers will be added.
            outlier_scale: Magnitude scale of the outliers.
        """
        self.dim = 5  # 5D function
        self._bounds = [(-5.0, 5.0)] * 5 # Domain bounds: [(x_i min, x_i max),]
        super().__init__(
            noise_std=noise_std, 
            negate=negate, 
            outlier_prob=outlier_prob, 
            outlier_scale=outlier_scale,
            outlier_std=outlier_std
        )
    
    def evaluate_true(self, X: Tensor) -> Tensor:
        r"""
        Compute the true function value.

        Args:
            X: A (batch_shape) x 5 tensor of input points.

        Returns:
            A tensor of function evaluations.
        """
        x1 = X[..., 0]
        x2 = X[..., 1]
        term1 = -20 * torch.exp(-0.2 * torch.sqrt(0.5 * (x1**2 + x2**2)))
        term2 = -torch.exp(0.5 * (torch.cos(2 * torch.pi * x1) + torch.cos(2 * torch.pi * x2)))
        result = term1 + term2 + 20 + torch.exp(torch.tensor(1.0, device=X.device, dtype=X.dtype))
        
        # return result
        return result.unsqueeze(-1)
    

class Hartmann6d(OutlierTestProblem):
    r"""
    Hartmann function in 6D with outliers:
        f(x_1, x_2, x_3, x_4, x_5, x_6) = -sum_{i=1}^4 a_i * exp(-sum_{j=1}^6 A_{ij} (x_j - P_{ij})^2),
        where a = [1.0, 1.2, 3.0, 3.2], A = [[10, 3, 17, 3.5, 1.7, 8], [0.05, 10, 17, 0.1, 8, 14], [3, 3.5, 1.7, 10, 17, 8], [17, 8, 0.05, 10, 0.1, 14]],
        P = [[0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886], [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991], [0.2348, 0.1451, 0.3522, 0.2883, 0.3047, 0.6650], [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381]],
        x_i ∈ [0, 1] for i = 1, 2, ..., 6.
    """

    def __init__(
        self, 
        noise_std: Optional[float] = None, 
        negate: bool = False,
        outlier_prob: Optional[float] = None,
        outlier_scale: float = 5.0,
        outlier_std: float = 1.0,
    ):
        r"""
        Args:
            noise_std: Standard deviation of the observation noise.
            negate: If True, negate the function.
            outlier_prob: Probability of an outlier occurring for each function evaluation. 
                          If None, no outliers will be added.
            outlier_scale: Magnitude scale of the outliers.
        """
    
        self.dim = 6  # 6D function
        self._bounds = [(0.0, 1.0)] * 6  # Domain bounds: [(x_i min, x_i max),]
        super().__init__(
            noise_std=noise_std, 
            negate=negate, 
            outlier_prob=outlier_prob, 
            outlier_scale=outlier_scale,
            outlier_std=outlier_std
        )
    
    def evaluate_true(self, X: Tensor) -> Tensor:
        r"""
        Compute the true function value.

        Args:
            X: A (batch_shape) x 6 tensor of input points.

        Returns:
            A tensor of function evaluations.
        """
        a = torch.tensor([1.0, 1.2, 3.0, 3.2], device=X.device, dtype=X.dtype)
        A = torch.tensor([
            [10, 3, 17, 3.5, 1.7, 8], 
            [0.05, 10, 17, 0.1, 8, 14], 
            [3, 3.5, 1.7, 10, 17, 8], 
            [17, 8, 0.05, 10, 0.1, 14]
        ], device=X.device, dtype=X.dtype)
        P = torch.tensor([
            [0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886], 
            [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991], 
            [0.2348, 0.1451, 0.3522, 0.2883, 0.3047, 0.6650], 
            [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381]
        ], device=X.device, dtype=X.dtype)
        
        result = -torch.sum(a * torch.exp(-torch.sum(A * (X.unsqueeze(-2) - P) ** 2, dim=-1)), dim=-1)
        
        # return result
        return result.unsqueeze(-1)



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

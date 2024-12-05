"""
1. 以下の実装を参考にして botorch model を作る
https://github.com/yucenli/bnn-bo/blob/main/models/laplace.py

2. ax で使用できるようにする
https://botorch.org/tutorials/custom_botorch_model_in_ax

3. 獲得関数は以下を使用する
https://botorch.org/api/acquisition.html#botorch.acquisition.monte_carlo.qNoisyExpectedImprovement
"""


import torch
import torch.nn as nn
import torch.optim as optim
import plotly.graph_objects as go
import numpy as np


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_units, activations):
        super(MLP, self).__init__()
        layers = []
        in_dim = input_dim

        for hidden_dim, activation in zip(hidden_units, activations):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(self.get_activation(activation))
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

    @staticmethod
    def get_activation(activation):
        if activation == "relu":
            return nn.ReLU()
        elif activation == "tanh":
            return nn.Tanh()
        elif activation == "sigmoid":
            return nn.Sigmoid()
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

# 真の関数 (ground truth)
def true_function(x):
    return 3 * np.sin(x) + x

# ラプラス近似のためのヘッセ行列近似を計算する関数
def hessian_diag(model, loss):
    # ヘッセ行列の対角成分を計算
    loss.backward(create_graph=True)
    diag_hessian = []
    for param in model.parameters():
        diag_hessian.append(param.grad.pow(2).detach())
    return diag_hessian

# ラプラス近似に基づく予測
def laplace_approximation_predict(model, X_test, diag_hessian, n_samples=100):
    sampled_preds = []

    for _ in range(n_samples):
        perturbed_model = MLP(**model_settings)
        for p_perturbed, p_orig, h_diag in zip(perturbed_model.parameters(), model.parameters(), diag_hessian):
            # ラプラス近似に基づいてガウス分布から重みをサンプリング
            std = torch.sqrt(h_diag)
            perturbed_model_param = p_orig + torch.randn_like(p_orig) * std
            p_perturbed.data.copy_(perturbed_model_param)
        
        sampled_preds.append(perturbed_model(X_test).detach())

    # すべてのサンプルから平均と標準偏差を計算
    preds = torch.stack(sampled_preds)
    mean_preds = preds.mean(0)
    std_preds = preds.std(0)
    
    return mean_preds, std_preds

# モデルのトレーニングと予測
def train_and_predict_with_laplace(model_settings, num_epochs=100):
    # データ準備
    torch.manual_seed(42)
    X_train = torch.randn(100, 1)  # 訓練データ
    y_train = true_function(X_train) + torch.randn(100, 1)  # ノイズを加えた出力

    model = MLP(**model_settings)  # モデルのインスタンス化
    criterion = nn.MSELoss()  # 損失関数
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # モデルのトレーニング
    for epoch in range(num_epochs):
        model.train()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # ラプラス近似のためにヘッセ行列の対角成分を計算
    model.eval()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    diag_hessian = hessian_diag(model, loss)

    # ラプラス近似に基づく予測
    X_test = torch.linspace(-3, 3, 100).unsqueeze(1)  # テストデータ
    mean_preds, std_preds = laplace_approximation_predict(model, X_test, diag_hessian)

    return X_train, y_train, X_test, mean_preds, std_preds


# プロット
def plot_results(X_train, y_train, X_test, mean_preds, std_preds, n_sigma=2):
    # データの変換
    X_train_np, y_train_np = X_train.numpy(), y_train.numpy()
    X_test_np, mean_preds_np = X_test.numpy(), mean_preds.numpy()
    y_true_np = true_function(X_test_np)
    std_preds_np = std_preds.numpy()

    # Plotlyによる可視化
    fig = go.Figure()

    # 訓練データ点
    fig.add_trace(go.Scatter(
        x=X_train_np.flatten(),
        y=y_train_np.flatten(),
        mode='markers',
        name='Training Data',
        marker=dict(color='blue', size=6)
    ))

    # 真の関数の軌跡
    fig.add_trace(go.Scatter(
        x=X_test_np.flatten(),
        y=y_true_np.flatten(),
        mode='lines',
        name='True Function',
        line=dict(color='green', width=2)
    ))

    # NN の予測の平均軌跡
    fig.add_trace(go.Scatter(
        x=X_test_np.flatten(),
        y=mean_preds_np.flatten(),
        mode='lines',
        name='NN Prediction (Mean)',
        line=dict(color='red', width=2)
    ))

    # 標準偏差の範囲をプロット (不確実性を表示)
    fig.add_trace(go.Scatter(
        x=np.concatenate([X_test_np.flatten(), X_test_np.flatten()[::-1]]),
        y=np.concatenate([mean_preds_np.flatten() - n_sigma * std_preds_np.flatten(), 
                          (mean_preds_np.flatten() + n_sigma * std_preds_np.flatten())[::-1]]),
        fill='toself',
        fillcolor='rgba(255, 0, 0, 0.2)',  # 塗りつぶしの色
        line=dict(color='rgba(255, 0, 0, 0)'),  # 標準偏差の範囲の線の色を透明に
        hoverinfo="skip",
        showlegend=False
    ))

    # グラフのレイアウト設定
    fig.update_layout(
        title="BNN with Laplace Approximation",
        xaxis_title="X",
        yaxis_title="y",
        width=800,
        height=600
    )

    fig.show()


if __name__ == "__main__":
    # モデルの設定
    model_settings = {
        "input_dim": 1,
        "output_dim": 1,
        "hidden_units": [128, 128, 128],
        "activations": ["tanh", "tanh", "tanh"]
    }

    # トレーニングとラプラス近似での予測
    X_train, y_train, X_test, mean_preds, std_preds = train_and_predict_with_laplace(model_settings)

    # 結果の可視化
    plot_results(X_train, y_train, X_test, mean_preds, std_preds, n_sigma=5)



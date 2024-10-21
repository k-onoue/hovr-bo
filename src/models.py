import torch
import torch.nn as nn
import torch.optim as optim
import plotly.graph_objects as go
import numpy as np

# MLPの定義
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_units, activations):
        super(MLP, self).__init__()
        layers = []
        in_dim = input_dim

        # 指定された hidden_units と activations に基づいて層を構築
        for hidden_dim, activation in zip(hidden_units, activations):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(self.get_activation(activation))
            in_dim = hidden_dim

        # 最後の出力層
        layers.append(nn.Linear(in_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

    @staticmethod
    def get_activation(activation):
        # 活性化関数の設定
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

# モデルのトレーニングと予測
def train_and_predict(model_settings, num_epochs=1000):
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

    # モデルの予測
    model.eval()
    with torch.no_grad():
        X_test = torch.linspace(-3, 3, 100).unsqueeze(1)  # テスト用データ
        y_pred = model(X_test)

    return X_train, y_train, X_test, y_pred

# プロット
def plot_results(X_train, y_train, X_test, y_pred):
    # データの変換
    X_train_np, y_train_np = X_train.numpy(), y_train.numpy()
    X_test_np, y_pred_np = X_test.numpy(), y_pred.numpy()
    y_true_np = true_function(X_test_np)

    # Plotlyによる可視化
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=X_train_np.flatten(),
        y=y_train_np.flatten(),
        mode='markers',
        name='Training Data',
        marker=dict(color='blue', size=6)
    ))

    fig.add_trace(go.Scatter(
        x=X_test_np.flatten(),
        y=y_true_np.flatten(),
        mode='lines',
        name='True Function',
        line=dict(color='green', width=2)
    ))

    fig.add_trace(go.Scatter(
        x=X_test_np.flatten(),
        y=y_pred_np.flatten(),
        mode='lines',
        name='NN Prediction',
        line=dict(color='red', width=2)
    ))

    fig.update_layout(
        title="MLP Regression with True Function and Predictions",
        xaxis_title="X",
        yaxis_title="y",
        legend=dict(x=0.8, y=0.1),
        width=800,
        height=600
    )

    fig.show()

if __name__ == "__main__":
    # モデルの設定
    model_settings = {
        "input_dim": 1,
        "output_dim": 1,
        "hidden_units": [10, 20],
        "activations": ["relu", "tanh"]
    }

    # トレーニングと予測
    X_train, y_train, X_test, y_pred = train_and_predict(model_settings)

    # 結果の可視化
    plot_results(X_train, y_train, X_test, y_pred)

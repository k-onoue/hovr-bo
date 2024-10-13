import numpy as np
import torch
import torch.nn as nn


# MLPモデルの定義
class MLP(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=100, output_dim=1, activation_func='relu'):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU() if activation_func == 'relu' else nn.Sigmoid(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU() if activation_func == 'relu' else nn.Sigmoid(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU() if activation_func == 'relu' else nn.Sigmoid(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.layers(x)


# HOVR正則化の定義
def hovr_regularization(model, x, k=2, q=2, num_points=10):
    # ランダムサンプルの生成
    x_min, x_max = x.min(0)[0], x.max(0)[0]
    random_points = torch.tensor(np.random.uniform(x_min.numpy(), x_max.numpy(), (num_points, x.shape[1])), 
                                 dtype=torch.float32, requires_grad=True)
    
    # モデルの出力と勾配の計算
    preds = model(random_points)
    grads = torch.autograd.grad(preds, random_points, torch.ones_like(preds), create_graph=True)[0]
    
    # HOVR項の計算
    hovr_term = 0.0
    for i in range(x.shape[1]):  # 各次元に対するk次導関数を計算
        grad_i = grads[:, i]
        temp_grad = grad_i
        for _ in range(k - 1):
            temp_grad = torch.autograd.grad(temp_grad, random_points, torch.ones_like(temp_grad), create_graph=True)[0][:, i]
        hovr_term += torch.sum(torch.abs(temp_grad) ** q)
    
    return hovr_term / x.shape[1]


# モデルの訓練
def train_model(X, y, model, optimizer, hovr_lambda=None, hovr_k=None, hovr_q=None, num_epochs=5000):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y.reshape(-1, 1), dtype=torch.float32)
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        preds = model(X_tensor)
        loss = nn.MSELoss()(preds, y_tensor)
        
        # HOVR正則化を追加
        if hovr_lambda is not None and hovr_k is not None and hovr_q is not None:
            hovr_term = hovr_regularization(model, X_tensor, k=hovr_k, q=hovr_q)
            loss += hovr_lambda * hovr_term

        loss.backward()
        optimizer.step()

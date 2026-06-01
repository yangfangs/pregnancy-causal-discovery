"""
NOTEARS-MLP 因果发现 (GPU 加速)
连续优化方法 + 非线性 MLP 扩展，利用 RTX 3080 Ti
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import OUTPUT_ROOT


class NotearsMLP(nn.Module):
    """NOTEARS with MLP for nonlinear causal discovery"""

    def __init__(self, n_features: int, hidden_dim: int = 10):
        super().__init__()
        self.n_features = n_features
        # 每个变量一个 MLP: X_j = f_j(PA(X_j)) + noise
        self.W1 = nn.Parameter(torch.randn(n_features, n_features, hidden_dim) * 0.01)
        self.W2 = nn.Parameter(torch.randn(n_features, hidden_dim, 1) * 0.01)
        self.bias1 = nn.Parameter(torch.zeros(n_features, hidden_dim))
        self.bias2 = nn.Parameter(torch.zeros(n_features, 1))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """X: (batch, n_features) -> X_hat: (batch, n_features)"""
        d = self.n_features
        batch = X.shape[0]
        # 逐变量计算: X_hat_j = W2[j] @ sigmoid(W1[j] @ X + bias1[j]) + bias2[j]
        X_hat = torch.zeros(batch, d, device=X.device)
        for j in range(d):
            # W1[j]: (d, h), X: (batch, d)
            h = X @ self.W1[j] + self.bias1[j]  # (batch, h)
            h = torch.sigmoid(h)
            X_hat[:, j] = (h @ self.W2[j]).squeeze(-1) + self.bias2[j, 0]  # (batch,)
        return X_hat

    def get_adj_matrix(self) -> torch.Tensor:
        """计算有效邻接矩阵: adj[i,j] 表示 i→j 的连接强度"""
        d = self.n_features
        adj = torch.zeros(d, d, device=self.W1.device)
        for j in range(d):
            # 变量 j 的输入权重: W1[j] 是 (d, h), W2[j] 是 (h, 1)
            # 输入 i 对输出 j 的贡献 = ||W1[j,i,:] * W2[j,:,0]||
            w = (self.W1[j] ** 2).sum(dim=1)  # (d,) — 每个输入的权重范数
            adj[:, j] = w  # adj[i,j] = 输入 i 对变量 j 的影响
        return adj


def _h_acyclicity(W: torch.Tensor) -> torch.Tensor:
    """DAG 无环约束: h(W) = tr(exp(W * W)) - d = 0"""
    d = W.shape[0]
    M = W * W
    exp_M = torch.matrix_exp(M)
    return torch.trace(exp_M) - d


def train_notears_mlp(data: np.ndarray, features: list,
                       hidden_dim: int = 10, lambda1: float = 0.05,
                       max_iter: int = 200, lr: float = 0.003,
                       rho_init: float = 1.0, rho_max: float = 1e16,
                       h_tol: float = 1e-8, device: str = "cuda"):
    """训练 NOTEARS-MLP 模型"""
    d = len(features)
    n = data.shape[0]

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    X = torch.tensor(data, dtype=torch.float32, device=device)

    model = NotearsMLP(d, hidden_dim).to(device)
    optimizer = Adam(model.parameters(), lr=lr)

    rho = rho_init
    alpha = 0.0  # augmented Lagrangian multiplier
    h_prev = np.inf

    for outer in range(10):  # augmented Lagrangian outer iterations
        for inner in range(max_iter):
            optimizer.zero_grad()

            X_hat = model(X)
            mse_loss = 0.5 / n * torch.sum((X - X_hat) ** 2)

            adj = model.get_adj_matrix()
            l1_penalty = lambda1 * torch.sum(torch.abs(adj))

            h = _h_acyclicity(adj)
            augmented = mse_loss + l1_penalty + alpha * h + 0.5 * rho * h * h

            augmented.backward()
            optimizer.step()

        with torch.no_grad():
            adj = model.get_adj_matrix()
            h_val = _h_acyclicity(adj).item()

        print(f"    outer={outer}, h={h_val:.6f}, rho={rho:.1f}")

        if h_val < h_tol:
            break
        if h_val > 0.25 * h_prev:
            rho = min(rho * 10, rho_max)
        alpha += rho * h_val
        h_prev = h_val

    # 提取邻接矩阵
    with torch.no_grad():
        W = model.get_adj_matrix().cpu().numpy()

    return W


def threshold_and_orient(W: np.ndarray, threshold: float = 0.2) -> np.ndarray:
    """阈值化并确保无环"""
    W_binary = (np.abs(W) > threshold).astype(int)
    # 简单去环: 保留更强的边
    for i in range(W_binary.shape[0]):
        for j in range(i + 1, W_binary.shape[1]):
            if W_binary[i, j] and W_binary[j, i]:
                if abs(W[i, j]) >= abs(W[j, i]):
                    W_binary[j, i] = 0
                else:
                    W_binary[i, j] = 0
    return W_binary


def bootstrap_notears(data: np.ndarray, features: list, n_bootstrap: int = 50,
                       threshold: float = 0.2, device: str = "cuda", **kwargs) -> np.ndarray:
    """Bootstrap NOTEARS-MLP，返回边频率矩阵"""
    n_samples = data.shape[0]
    n_features = len(features)
    freq_matrix = np.zeros((n_features, n_features))

    for b in range(n_bootstrap):
        idx = np.random.choice(n_samples, size=n_samples, replace=True)
        boot_data = data[idx]
        try:
            W = train_notears_mlp(boot_data, features, device=device, **kwargs)
            directed = threshold_and_orient(W, threshold=threshold)
            freq_matrix += directed
        except Exception as e:
            print(f"    bootstrap {b+1} failed: {e}")
            continue
        if (b + 1) % 10 == 0:
            print(f"    NOTEARS bootstrap {b+1}/{n_bootstrap}")

    freq_matrix /= n_bootstrap
    return freq_matrix


def run(datasets: dict, n_bootstrap: int = 50, device: str = "cuda"):
    """对每个 trimester 运行 NOTEARS-MLP"""
    out_dir = OUTPUT_ROOT / "causal_graphs"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for tri_name, tri_info in datasets.items():
        features = tri_info["features"]
        imp_df = tri_info["imputed"][0]

        # 采样以加速（NOTEARS 对大样本慢）
        max_n = 10000
        if len(imp_df) > max_n:
            sample_df = imp_df.sample(n=max_n, random_state=42)
        else:
            sample_df = imp_df
        data = sample_df[features].values.astype(np.float64)

        print(f"\n=== NOTEARS-MLP {tri_name} ({len(features)} features, n={len(sample_df)}) ===")

        # 单次运行（不做 bootstrap 先看效果）
        W = train_notears_mlp(data, features, device=device)
        directed = threshold_and_orient(W, threshold=0.2)

        # 保存
        w_df = pd.DataFrame(W, index=features, columns=features)
        w_df.to_csv(out_dir / f"{tri_name}_all_notears_weights.csv")

        d_df = pd.DataFrame(directed, index=features, columns=features)
        d_df.to_csv(out_dir / f"{tri_name}_all_notears_freq.csv")

        all_results[tri_name] = {"weights": W, "directed": directed, "features": features}
        n_edges = directed.sum()
        print(f"  {tri_name}: {n_edges} directed edges")

    return all_results


if __name__ == "__main__":
    print("请通过 pipeline/run_all.py 运行")

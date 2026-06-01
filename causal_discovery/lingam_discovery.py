"""
LiNGAM 因果发现
利用非高斯性识别因果方向 (DirectLiNGAM)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lingam

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import OUTPUT_ROOT


def run_direct_lingam(data: np.ndarray, features: list) -> np.ndarray:
    """运行 DirectLiNGAM，返回邻接矩阵"""
    model = lingam.DirectLiNGAM()
    model.fit(data)
    return model.adjacency_matrix_  # B[i,j] != 0 means j -> i


def bootstrap_lingam(data: np.ndarray, features: list, n_bootstrap: int = 100,
                      threshold: float = 0.05) -> np.ndarray:
    """Bootstrap LiNGAM，返回边频率矩阵"""
    n_samples, n_features = data.shape
    freq_matrix = np.zeros((n_features, n_features))

    for b in range(n_bootstrap):
        idx = np.random.choice(n_samples, size=n_samples, replace=True)
        boot_data = data[idx]
        try:
            B = run_direct_lingam(boot_data, features)
            # B[i,j] != 0 means j -> i, 转为 directed[j,i] = 1
            directed = (np.abs(B) > threshold).astype(int).T
            freq_matrix += directed
        except Exception:
            continue
        if (b + 1) % 20 == 0:
            print(f"    LiNGAM bootstrap {b+1}/{n_bootstrap}")

    freq_matrix /= n_bootstrap
    return freq_matrix


def run(datasets: dict, n_bootstrap: int = 100):
    """对每个 trimester 运行 LiNGAM"""
    out_dir = OUTPUT_ROOT / "causal_graphs"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for tri_name, tri_info in datasets.items():
        features = tri_info["features"]
        imp_df = tri_info["imputed"][0]

        # 采样
        max_n = 10000
        if len(imp_df) > max_n:
            sample_df = imp_df.sample(n=max_n, random_state=42)
        else:
            sample_df = imp_df
        data = sample_df[features].values.astype(np.float64)

        print(f"\n=== LiNGAM {tri_name} ({len(features)} features, n={len(sample_df)}) ===")

        freq_matrix = bootstrap_lingam(data, features, n_bootstrap=n_bootstrap)

        freq_df = pd.DataFrame(freq_matrix, index=features, columns=features)
        freq_df.to_csv(out_dir / f"{tri_name}_all_lingam_freq.csv")

        all_results[tri_name] = {"freq_matrix": freq_matrix, "features": features}
        n_edges = (freq_matrix > 0.5).sum()
        print(f"  {tri_name}: {n_edges} stable edges (freq > 0.5)")

    return all_results


if __name__ == "__main__":
    print("请通过 pipeline/run_all.py 运行")

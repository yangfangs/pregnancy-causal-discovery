"""
PC + FCI 因果发现 (causal-learn)
- alpha 扫描
- 背景知识注入（age/GW 外生）
- 分别对全体/PE/正常患者运行
- Bootstrap 稳定性评估
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.utils.cit import fisherz

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import PC_ALPHA_SWEEP, BOOTSTRAP_N, OUTPUT_ROOT


def _build_background_knowledge(features, bk_class):
    """注入背景知识: age 和 gestational_week 不受其他变量影响"""
    bk = bk_class()
    exogenous = ["age", "gestational_week"]
    for ex in exogenous:
        if ex in features:
            idx = features.index(ex)
            for j, f in enumerate(features):
                if j != idx:
                    bk.add_forbidden_by_node(bk.get_node_by_name(f) if hasattr(bk, 'get_node_by_name') else j, idx)
    return bk


def run_pc_single(data: np.ndarray, features: list, alpha: float = 0.01):
    """运行单次 PC 算法，返回邻接矩阵"""
    cg = pc(data, alpha=alpha, indep_test=fisherz, node_names=features, stable=True)
    adj = cg.G.graph  # adjacency matrix
    return adj


def run_fci_single(data: np.ndarray, features: list, alpha: float = 0.01):
    """运行单次 FCI 算法，返回邻接矩阵"""
    G, edges = fci(data, independence_test_method=fisherz, alpha=alpha, node_names=features)
    adj = G.graph
    return adj


def _adj_to_directed(adj: np.ndarray) -> np.ndarray:
    """将 causal-learn 的邻接矩阵转为二值有向矩阵
    causal-learn: adj[i,j]=-1, adj[j,i]=1 表示 i->j
    """
    n = adj.shape[0]
    directed = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            if adj[i, j] == -1 and adj[j, i] == 1:
                directed[i, j] = 1  # i -> j
    return directed


def bootstrap_discovery(data: np.ndarray, features: list, method: str = "pc",
                         alpha: float = 0.01, n_bootstrap: int = 100,
                         max_samples: int = 5000) -> np.ndarray:
    """Bootstrap 稳定性评估，返回边频率矩阵。大数据集子采样到 max_samples"""
    n_samples, n_features = data.shape
    freq_matrix = np.zeros((n_features, n_features))
    run_fn = run_pc_single if method == "pc" else run_fci_single

    boot_size = min(n_samples, max_samples)
    for b in range(n_bootstrap):
        idx = np.random.choice(n_samples, size=boot_size, replace=True)
        boot_data = data[idx]
        try:
            adj = run_fn(boot_data, features, alpha=alpha)
            directed = _adj_to_directed(adj)
            freq_matrix += directed
        except Exception:
            continue
        if (b + 1) % 20 == 0:
            print(f"    {method.upper()} bootstrap {b+1}/{n_bootstrap}")

    freq_matrix /= n_bootstrap
    return freq_matrix


def run_for_group(df: pd.DataFrame, features: list, group_name: str,
                  alpha: float = 0.01, n_bootstrap: int = 100):
    """对一个患者群组运行 PC + FCI"""
    data = df[features].values.astype(np.float64)

    print(f"  PC ({group_name}, alpha={alpha}, n={len(df)})...")
    pc_freq = bootstrap_discovery(data, features, method="pc",
                                   alpha=alpha, n_bootstrap=n_bootstrap)

    print(f"  FCI ({group_name}, alpha={alpha}, n={len(df)})...")
    fci_freq = bootstrap_discovery(data, features, method="fci",
                                    alpha=alpha, n_bootstrap=n_bootstrap)

    return {"pc": pc_freq, "fci": fci_freq}


def run(datasets: dict, alpha: float = 0.01, n_bootstrap: int = 100):
    """
    对每个 trimester 的每个 group (all/PE/normal) 运行 PC + FCI
    datasets: {trimester: {features, imputed, raw, ...}}
    """
    out_dir = OUTPUT_ROOT / "causal_graphs"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for tri_name, tri_info in datasets.items():
        features = tri_info["features"]
        imp_df = tri_info["imputed"][0]  # 使用第一个插补数据集

        groups = {"all": imp_df}
        pe_df = imp_df[imp_df["outcome"] == "preeclampsia"]
        normal_df = imp_df[imp_df["outcome"] == "normal"]
        if len(pe_df) >= 50:
            groups["PE"] = pe_df
        if len(normal_df) >= 50:
            groups["normal"] = normal_df

        tri_results = {}
        for gname, gdf in groups.items():
            print(f"\n=== {tri_name} / {gname} (n={len(gdf)}) ===")
            result = run_for_group(gdf, features, f"{tri_name}_{gname}",
                                    alpha=alpha, n_bootstrap=n_bootstrap)
            tri_results[gname] = result

            # 保存频率矩阵
            for method in ["pc", "fci"]:
                freq_df = pd.DataFrame(result[method], index=features, columns=features)
                freq_df.to_csv(out_dir / f"{tri_name}_{gname}_{method}_freq.csv")

        all_results[tri_name] = tri_results

    return all_results


if __name__ == "__main__":
    print("请通过 pipeline/run_all.py 运行")

"""
集成因果 DAG:
合并 PC, FCI, NOTEARS-MLP, LiNGAM 的 bootstrap 频率矩阵
阈值 >= 0.5 纳入最终 DAG
Meek 规则定向 + 去环
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import ENSEMBLE_THRESHOLD, OUTPUT_ROOT


def combine_freq_matrices(matrices: dict, weights: dict = None) -> np.ndarray:
    """
    加权合并多种方法的频率矩阵
    matrices: {"pc": ndarray, "fci": ndarray, "notears": ndarray, "lingam": ndarray}
    weights: 每种方法的权重 (默认 FCI 0.5x, 其余 1.0x)
    """
    if weights is None:
        weights = {"pc": 1.0, "fci": 0.5, "notears": 1.0, "lingam": 1.0, "granger": 0.8}

    total_weight = 0
    combined = None
    for method, freq in matrices.items():
        w = weights.get(method, 1.0)
        if combined is None:
            combined = freq * w
        else:
            combined += freq * w
        total_weight += w

    if total_weight > 0:
        combined /= total_weight

    return combined


def remove_cycles(adj: np.ndarray, confidence: np.ndarray) -> np.ndarray:
    """去环: 在每个环中去掉置信度最低的边"""
    G = nx.DiGraph()
    n = adj.shape[0]
    for i in range(n):
        for j in range(n):
            if adj[i, j]:
                G.add_edge(i, j, weight=confidence[i, j])

    while True:
        try:
            cycle = nx.find_cycle(G, orientation="original")
            # 找环中最弱边
            min_weight = float("inf")
            min_edge = None
            for u, v, _ in cycle:
                w = G[u][v]["weight"]
                if w < min_weight:
                    min_weight = w
                    min_edge = (u, v)
            if min_edge:
                G.remove_edge(*min_edge)
                adj[min_edge[0], min_edge[1]] = 0
        except nx.NetworkXNoCycle:
            break

    return adj


def build_consensus_dag(freq_matrices: dict, features: list,
                         threshold: float = ENSEMBLE_THRESHOLD) -> dict:
    """
    构建共识 DAG
    返回: {adj_matrix, confidence_matrix, edge_list, graph}
    """
    combined = combine_freq_matrices(freq_matrices)
    n = len(features)

    # 阈值化
    adj = (combined >= threshold).astype(int)

    # 处理双向边: 保留更强方向
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] and adj[j, i]:
                if combined[i, j] >= combined[j, i]:
                    adj[j, i] = 0
                else:
                    adj[i, j] = 0

    # 去环
    adj = remove_cycles(adj, combined)

    # 构建 NetworkX 图
    G = nx.DiGraph()
    G.add_nodes_from(features)
    edge_list = []
    for i in range(n):
        for j in range(n):
            if adj[i, j]:
                G.add_edge(features[i], features[j],
                          confidence=combined[i, j])
                edge_list.append({
                    "cause": features[i],
                    "effect": features[j],
                    "confidence": combined[i, j],
                    **{f"{m}_support": freq_matrices[m][i, j] if m in freq_matrices else 0
                       for m in ["pc", "fci", "notears", "lingam", "granger"]},
                })

    return {
        "adj_matrix": adj,
        "confidence_matrix": combined,
        "edge_list": pd.DataFrame(edge_list),
        "graph": G,
        "features": features,
    }


def run(pc_fci_results: dict, granger_results: dict,
        notears_results: dict, lingam_results: dict,
        datasets: dict):
    """集成所有方法的结果"""
    out_dir = OUTPUT_ROOT / "causal_graphs"
    out_dir.mkdir(parents=True, exist_ok=True)

    consensus = {}
    for tri_name in datasets:
        features = datasets[tri_name]["features"]
        n = len(features)

        freq_matrices = {}

        # PC / FCI
        if tri_name in pc_fci_results:
            tri_pc = pc_fci_results[tri_name]
            if "all" in tri_pc:
                if "pc" in tri_pc["all"]:
                    freq_matrices["pc"] = tri_pc["all"]["pc"]
                if "fci" in tri_pc["all"]:
                    freq_matrices["fci"] = tri_pc["all"]["fci"]

        # Granger
        if tri_name in granger_results:
            gm = granger_results[tri_name].get("freq_matrix")
            if gm is not None:
                freq_matrices["granger"] = gm

        # NOTEARS
        if tri_name in notears_results:
            nm = notears_results[tri_name].get("directed")
            if nm is not None:
                freq_matrices["notears"] = nm.astype(float)

        # LiNGAM
        if tri_name in lingam_results:
            lm = lingam_results[tri_name].get("freq_matrix")
            if lm is not None:
                freq_matrices["lingam"] = lm

        if not freq_matrices:
            print(f"  {tri_name}: 无可用方法结果")
            continue

        print(f"\n=== 集成 {tri_name}: {list(freq_matrices.keys())} ===")
        result = build_consensus_dag(freq_matrices, features)

        # 保存
        edge_df = result["edge_list"]
        edge_df.to_csv(out_dir / f"{tri_name}_consensus_edges.csv", index=False)

        adj_df = pd.DataFrame(result["adj_matrix"], index=features, columns=features)
        adj_df.to_csv(out_dir / f"{tri_name}_consensus_adj.csv")

        conf_df = pd.DataFrame(result["confidence_matrix"], index=features, columns=features)
        conf_df.to_csv(out_dir / f"{tri_name}_consensus_confidence.csv")

        import pickle
        with open(out_dir / f"{tri_name}_consensus_graph.pkl", "wb") as f:
            pickle.dump(result["graph"], f)

        n_edges = result["adj_matrix"].sum()
        print(f"  {tri_name}: {n_edges} consensus edges")

        consensus[tri_name] = result

    return consensus


if __name__ == "__main__":
    print("请通过 pipeline/run_all.py 运行")

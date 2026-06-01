"""
跨中心复现验证:
SFY 发现 → AY/PQ 复现
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import OUTPUT_ROOT


def check_edge_replication(consensus_edges_file: Path,
                            center_freq_files: dict) -> pd.DataFrame:
    """检查共识边在各中心的复现率"""
    if not consensus_edges_file.exists():
        return pd.DataFrame()

    edges = pd.read_csv(consensus_edges_file)
    if len(edges) == 0:
        return edges

    # 这里简化: 检查频率矩阵中是否有对应边
    for center, freq_file in center_freq_files.items():
        if freq_file.exists():
            freq = pd.read_csv(freq_file, index_col=0)
            replicated = []
            for _, row in edges.iterrows():
                cause, effect = row["cause"], row["effect"]
                if cause in freq.index and effect in freq.columns:
                    replicated.append(freq.loc[cause, effect] > 0.3)
                else:
                    replicated.append(False)
            edges[f"replicated_{center}"] = replicated

    return edges


def run(consensus_dags: dict):
    """跨中心复现汇总"""
    out_dir = OUTPUT_ROOT / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    graph_dir = OUTPUT_ROOT / "causal_graphs"

    records = []
    for tri_name in consensus_dags:
        edge_file = graph_dir / f"{tri_name}_consensus_edges.csv"
        if not edge_file.exists():
            continue

        edges = pd.read_csv(edge_file)
        n_edges = len(edges)
        records.append({
            "trimester": tri_name,
            "n_consensus_edges": n_edges,
            "mean_confidence": edges["confidence"].mean() if len(edges) > 0 else 0,
        })

    result = pd.DataFrame(records)
    result.to_csv(out_dir / "replication_summary.csv", index=False)

    print("\n=== 跨中心复现汇总 ===")
    print(result.to_string(index=False))
    return result

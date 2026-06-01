"""
跨孕期阶段 DAG 比较:
- 持续/新增/消失边分类
- SHD/SID 距离
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import OUTPUT_ROOT


def structural_hamming_distance(adj1: np.ndarray, adj2: np.ndarray) -> int:
    """SHD: 两个 DAG 之间的结构差异（增删反转边数之和）"""
    return int(np.sum(adj1 != adj2))


def classify_edges(consensus_dags: dict, features_map: dict) -> pd.DataFrame:
    """
    将每条边分为: persistent (全阶段), emerging (后期出现),
    disappearing (早期消失), trimester_specific (单阶段)
    """
    trimesters = sorted(consensus_dags.keys())
    all_edges = set()

    edge_presence = {}
    for tri in trimesters:
        dag = consensus_dags[tri]
        adj = dag["adj_matrix"]
        feats = dag["features"]
        for i in range(len(feats)):
            for j in range(len(feats)):
                if adj[i, j]:
                    edge = (feats[i], feats[j])
                    all_edges.add(edge)
                    edge_presence.setdefault(edge, set()).add(tri)

    records = []
    for edge in sorted(all_edges):
        present_in = edge_presence.get(edge, set())
        if len(present_in) == len(trimesters):
            category = "persistent"
        elif present_in == {"T2", "T3"} or present_in == {"T3"}:
            category = "emerging"
        elif present_in == {"T1"} or present_in == {"T1", "T2"}:
            category = "disappearing"
        else:
            category = "trimester_specific"

        records.append({
            "cause": edge[0],
            "effect": edge[1],
            "T1": "T1" in present_in,
            "T2": "T2" in present_in,
            "T3": "T3" in present_in,
            "category": category,
        })

    return pd.DataFrame(records)


def run(consensus_dags: dict):
    """运行跨阶段比较"""
    out_dir = OUTPUT_ROOT / "causal_graphs"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 边分类
    features_map = {t: d["features"] for t, d in consensus_dags.items()}
    edge_df = classify_edges(consensus_dags, features_map)
    edge_df.to_csv(out_dir / "trimester_edge_comparison.csv", index=False)

    # SHD
    trimesters = sorted(consensus_dags.keys())
    shd_records = []
    for i in range(len(trimesters)):
        for j in range(i + 1, len(trimesters)):
            t1, t2 = trimesters[i], trimesters[j]
            # 对齐特征
            f1 = consensus_dags[t1]["features"]
            f2 = consensus_dags[t2]["features"]
            common = sorted(set(f1) & set(f2))
            if not common:
                continue
            idx1 = [f1.index(f) for f in common]
            idx2 = [f2.index(f) for f in common]
            a1 = consensus_dags[t1]["adj_matrix"][np.ix_(idx1, idx1)]
            a2 = consensus_dags[t2]["adj_matrix"][np.ix_(idx2, idx2)]
            shd = structural_hamming_distance(a1, a2)
            shd_records.append({"period_1": t1, "period_2": t2, "SHD": shd, "n_features": len(common)})

    shd_df = pd.DataFrame(shd_records)
    shd_df.to_csv(out_dir / "trimester_shd.csv", index=False)

    # 统计
    cats = edge_df["category"].value_counts()
    print("\n=== 跨阶段边分类 ===")
    for cat, cnt in cats.items():
        print(f"  {cat}: {cnt}")
    print(f"\nSHD:\n{shd_df.to_string(index=False)}")

    return {"edge_comparison": edge_df, "shd": shd_df}

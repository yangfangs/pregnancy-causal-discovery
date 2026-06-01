"""
因果级联链识别:
- DAG 最长路径枚举 (≥3 节点)
- 跨器官系统过滤
- 路径强度评分
"""

import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.feature_groups import feature_to_group, CLINICAL_GROUPS
from config.settings import OUTPUT_ROOT


def enumerate_paths(G: nx.DiGraph, min_length: int = 3, max_length: int = 8) -> list:
    """枚举 DAG 中长度 >= min_length 的所有有向路径"""
    paths = []
    for source in G.nodes():
        for target in G.nodes():
            if source == target:
                continue
            for path in nx.all_simple_paths(G, source, target,
                                             cutoff=max_length):
                if len(path) >= min_length:
                    paths.append(path)
    return paths


def path_strength(path: list, G: nx.DiGraph) -> float:
    """路径强度 = 各边置信度的乘积"""
    strength = 1.0
    for i in range(len(path) - 1):
        edge_data = G.get_edge_data(path[i], path[i + 1])
        if edge_data and "confidence" in edge_data:
            strength *= edge_data["confidence"]
        else:
            strength *= 0.5  # 默认
    return strength


def count_system_crossings(path: list) -> int:
    """计算路径跨越的器官系统数"""
    systems = [feature_to_group(node) for node in path]
    return len(set(systems)) - 1


def filter_clinical_paths(paths: list, G: nx.DiGraph,
                           min_crossings: int = 1) -> pd.DataFrame:
    """筛选临床有意义的路径"""
    records = []
    for path in paths:
        crossings = count_system_crossings(path)
        if crossings < min_crossings:
            continue

        strength = path_strength(path, G)
        systems = [feature_to_group(node) for node in path]

        records.append({
            "path": " → ".join(path),
            "length": len(path),
            "strength": strength,
            "n_system_crossings": crossings,
            "systems": " → ".join(systems),
            "start_system": systems[0],
            "end_system": systems[-1],
            "nodes": path,
        })

    df = pd.DataFrame(records)
    if len(df) > 0:
        df = df.sort_values("strength", ascending=False)
    return df


def run(consensus_dags: dict):
    """对每个 trimester 的共识 DAG 识别因果级联"""
    out_dir = OUTPUT_ROOT / "cascades"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_cascades = {}
    for tri_name, dag_info in consensus_dags.items():
        G = dag_info["graph"]
        print(f"\n=== 级联识别 {tri_name} ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges) ===")

        paths = enumerate_paths(G, min_length=3)
        print(f"  原始路径: {len(paths)}")

        cascade_df = filter_clinical_paths(paths, G, min_crossings=1)
        print(f"  临床路径 (跨系统≥1): {len(cascade_df)}")

        if len(cascade_df) > 0:
            cascade_df.to_csv(out_dir / f"{tri_name}_cascades.csv", index=False)
            print(f"  Top 5 路径:")
            for _, row in cascade_df.head(5).iterrows():
                print(f"    {row['path']} (strength={row['strength']:.3f}, systems: {row['systems']})")

        all_cascades[tri_name] = cascade_df

    return all_cascades

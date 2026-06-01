"""
出版级 DAG 可视化:
- 三阶段 DAG 并排
- 节点按器官系统着色
- 边宽 = 置信度
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.feature_groups import CLINICAL_GROUPS, feature_to_color, feature_to_group
from config.settings import OUTPUT_ROOT
from visualization.style import setup_lancet_style, get_figure_size


def draw_dag(G: nx.DiGraph, ax: plt.Axes, title: str = "",
             pos: dict = None, highlight_edges: set = None):
    """在指定 axes 上绘制 DAG"""
    if pos is None:
        try:
            pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
        except Exception:
            pos = nx.spring_layout(G, seed=42, k=2)

    # 节点颜色
    node_colors = [feature_to_color(n) for n in G.nodes()]

    # 边宽度和颜色
    edge_widths = []
    edge_colors = []
    for u, v in G.edges():
        conf = G[u][v].get("confidence", 0.5)
        edge_widths.append(max(0.5, conf * 3))
        if highlight_edges and (u, v) in highlight_edges:
            edge_colors.append("#E64B35")  # 红色高亮
        else:
            edge_colors.append("#333333")

    # 绘制
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                           node_size=300, edgecolors="#333333", linewidths=0.5)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=6, font_family="sans-serif")
    nx.draw_networkx_edges(G, pos, ax=ax, width=edge_widths,
                           edge_color=edge_colors, arrows=True,
                           arrowsize=8, arrowstyle="-|>",
                           connectionstyle="arc3,rad=0.1", alpha=0.8)

    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.axis("off")
    return pos


def plot_trimester_dags(consensus_dags: dict):
    """生成三阶段 DAG 并排图"""
    setup_lancet_style()
    out_dir = OUTPUT_ROOT / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    trimesters = sorted(consensus_dags.keys())
    n_panels = len(trimesters)
    if n_panels == 0:
        return

    fig, axes = plt.subplots(1, n_panels, figsize=get_figure_size("double"))
    if n_panels == 1:
        axes = [axes]

    # 收集所有节点以统一布局
    all_nodes = set()
    for dag_info in consensus_dags.values():
        all_nodes.update(dag_info["graph"].nodes())

    # 创建统一布局
    union_G = nx.DiGraph()
    union_G.add_nodes_from(all_nodes)
    for dag_info in consensus_dags.values():
        union_G.add_edges_from(dag_info["graph"].edges())
    try:
        shared_pos = nx.nx_agraph.graphviz_layout(union_G, prog="dot")
    except Exception:
        shared_pos = nx.spring_layout(union_G, seed=42, k=2)

    titles = {"T1": "T1 (0-14w)", "T2": "T2 (14-28w)", "T3": "T3 (28-45w)"}
    for ax, tri in zip(axes, trimesters):
        G = consensus_dags[tri]["graph"]
        # 过滤 pos 到当前图的节点
        pos = {n: shared_pos[n] for n in G.nodes() if n in shared_pos}
        draw_dag(G, ax, title=titles.get(tri, tri), pos=pos)

    # 图例
    legend_handles = []
    for gname, ginfo in CLINICAL_GROUPS.items():
        legend_handles.append(
            mpatches.Patch(color=ginfo["color"], label=ginfo["label"])
        )
    fig.legend(handles=legend_handles, loc="lower center", ncol=4, fontsize=7,
               frameon=False, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("Trimester-Stratified Causal DAGs of Pregnancy Lab Indicators",
                 fontsize=11, fontweight="bold", y=1.02)
    plt.tight_layout()

    for fmt in ["pdf", "png"]:
        fig.savefig(out_dir / f"trimester_dags.{fmt}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"DAG 图已保存到 {out_dir}/trimester_dags.pdf")


def run(consensus_dags: dict):
    plot_trimester_dags(consensus_dags)

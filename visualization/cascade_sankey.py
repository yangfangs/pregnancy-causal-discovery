"""
因果级联 Sankey 图 (使用 matplotlib 实现)
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.feature_groups import feature_to_color, feature_to_group, CLINICAL_GROUPS
from config.settings import OUTPUT_ROOT
from visualization.style import setup_lancet_style, get_figure_size


def plot_cascade_flow(scored_cascades: dict, top_n: int = 10):
    """为每个 trimester 绘制级联流程图（简化 Sankey）"""
    setup_lancet_style()
    out_dir = OUTPUT_ROOT / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    for tri_name, df in scored_cascades.items():
        if df is None or len(df) == 0:
            continue

        top = df.head(top_n)
        fig, ax = plt.subplots(figsize=get_figure_size("single"))

        y_pos = 0
        for _, row in top.iterrows():
            path_str = row["path"]
            nodes = path_str.split(" → ")
            strength = row.get("composite_score", row.get("strength", 0.5))

            # 绘制每条路径为一行箭头
            x_positions = np.linspace(0.1, 0.9, len(nodes))
            for i, (x, node) in enumerate(zip(x_positions, nodes)):
                color = feature_to_color(node)
                ax.scatter(x, y_pos, s=200, c=color, zorder=3, edgecolors="#333", linewidths=0.5)
                ax.annotate(node, (x, y_pos), textcoords="offset points",
                           xytext=(0, -12), ha="center", fontsize=5)
                if i < len(nodes) - 1:
                    ax.annotate("", xy=(x_positions[i + 1] - 0.02, y_pos),
                               xytext=(x + 0.02, y_pos),
                               arrowprops=dict(arrowstyle="->", color="#666",
                                              lw=max(0.5, strength * 2)))

            # 强度标签
            ax.text(0.95, y_pos, f"{strength:.2f}", fontsize=6, va="center")
            y_pos -= 1

        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(y_pos - 0.5, 1)
        ax.axis("off")
        ax.set_title(f"Top Causal Cascades - {tri_name}", fontsize=10, fontweight="bold")

        # 图例
        handles = [mpatches.Patch(color=g["color"], label=g["label"])
                   for g in CLINICAL_GROUPS.values()]
        ax.legend(handles=handles, loc="lower center", ncol=4, fontsize=6,
                  frameon=False, bbox_to_anchor=(0.5, -0.05))

        plt.tight_layout()
        for fmt in ["pdf", "png"]:
            fig.savefig(out_dir / f"cascades_{tri_name}.{fmt}", dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"级联图已保存: cascades_{tri_name}")


def run(scored_cascades: dict):
    plot_cascade_flow(scored_cascades)

"""
跨阶段 DAG 演变对比图:
- Panel A: 边分类条形图 (persistent/emerging/disappearing)
- Panel B: SHD 对比
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import OUTPUT_ROOT
from visualization.style import setup_lancet_style, get_figure_size, LANCET_COLORS


def plot_trimester_comparison(edge_comparison: pd.DataFrame, shd_df: pd.DataFrame):
    """绘制跨阶段对比图"""
    setup_lancet_style()
    out_dir = OUTPUT_ROOT / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=get_figure_size("single"))

    # Panel A: 边分类
    if len(edge_comparison) > 0:
        cats = edge_comparison["category"].value_counts()
        colors_map = {
            "persistent": LANCET_COLORS["blue"],
            "emerging": LANCET_COLORS["red"],
            "disappearing": LANCET_COLORS["green"],
            "trimester_specific": LANCET_COLORS["purple"],
        }
        bars = ax1.bar(cats.index, cats.values,
                       color=[colors_map.get(c, LANCET_COLORS["navy"]) for c in cats.index])
        ax1.set_ylabel("Number of Edges", fontsize=9)
        ax1.set_title("A. Edge Classification", fontsize=10, fontweight="bold")
        ax1.tick_params(axis="x", rotation=30)

        # 添加数值标签
        for bar, val in zip(bars, cats.values):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    str(val), ha="center", fontsize=7)

    # Panel B: SHD
    if len(shd_df) > 0:
        labels = [f"{r['period_1']}→{r['period_2']}" for _, r in shd_df.iterrows()]
        ax2.bar(labels, shd_df["SHD"], color=LANCET_COLORS["navy"])
        ax2.set_ylabel("Structural Hamming Distance", fontsize=9)
        ax2.set_title("B. DAG Structural Change", fontsize=10, fontweight="bold")

    plt.tight_layout()
    for fmt in ["pdf", "png"]:
        fig.savefig(out_dir / f"trimester_comparison.{fmt}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("跨阶段对比图已保存: trimester_comparison")


def run(edge_comparison, shd_df):
    if edge_comparison is not None and len(edge_comparison) > 0:
        plot_trimester_comparison(edge_comparison, shd_df)

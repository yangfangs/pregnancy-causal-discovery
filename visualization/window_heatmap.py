"""
ATT 热力图:
行 = 干预指标, 列 = 孕周窗口
核心图: 直接回答"何时干预最有效"
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import OUTPUT_ROOT
from visualization.style import setup_lancet_style, get_figure_size


def plot_att_heatmap(window_results: pd.DataFrame, outcome: str = "is_preeclampsia"):
    """绘制 ATT 热力图"""
    setup_lancet_style()
    out_dir = OUTPUT_ROOT / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = window_results[window_results["outcome"] == outcome].copy()
    if len(df) == 0:
        print(f"  无 {outcome} 窗口结果")
        return

    # 构建 pivot 表
    pivot = df.pivot_table(index="treatment", columns="gw_label", values="att", aggfunc="mean")

    # 排序列（按孕周顺序）
    col_order = sorted(pivot.columns, key=lambda x: int(x.split("-")[0]))
    pivot = pivot.reindex(columns=col_order)

    fig, ax = plt.subplots(figsize=get_figure_size("single"))

    # 使用发散色图: 蓝=保护效应(负ATT), 红=风险增加
    vmax = max(abs(pivot.min().min()), abs(pivot.max().max()), 0.01)
    sns.heatmap(pivot, ax=ax, cmap="RdBu_r", center=0, vmin=-vmax, vmax=vmax,
                annot=True, fmt=".3f", linewidths=0.5, linecolor="white",
                cbar_kws={"label": "ATT", "shrink": 0.8})

    ax.set_xlabel("Gestational Week Window", fontsize=9)
    ax.set_ylabel("Lab Abnormality (Treatment)", fontsize=9)

    outcome_label = {
        "is_preeclampsia": "Preeclampsia",
        "is_hypertension": "Hypertension",
        "is_gdm": "GDM",
    }.get(outcome, outcome)
    ax.set_title(f"Average Treatment Effect on {outcome_label}\nby Gestational Week Window",
                 fontsize=10, fontweight="bold")

    plt.tight_layout()
    for fmt in ["pdf", "png"]:
        fig.savefig(out_dir / f"att_heatmap_{outcome}.{fmt}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"ATT 热力图已保存: att_heatmap_{outcome}")


def run(window_results: pd.DataFrame):
    """为每种结局生成热力图"""
    if window_results is None or len(window_results) == 0:
        print("无窗口结果可绘图")
        return

    for outcome in ["is_preeclampsia", "is_hypertension", "is_gdm"]:
        plot_att_heatmap(window_results, outcome)

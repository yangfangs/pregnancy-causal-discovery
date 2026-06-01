"""
稳健性检验汇总可视化:
- Panel A: ATT vs IV vs RDD 三种方法效应对比 (forest plot)
- Panel B: RDD 断点图 (最显著的治疗变量)
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import OUTPUT_ROOT
from visualization.style import setup_lancet_style, get_figure_size, LANCET_COLORS


def plot_method_comparison(att_df: pd.DataFrame, iv_df: pd.DataFrame, rdd_df: pd.DataFrame):
    """Forest plot: 同一治疗变量下三种方法的效应估计对比"""
    setup_lancet_style()
    out_dir = OUTPUT_ROOT / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 收集 PE 全样本结果
    records = []

    # ATT (全样本均值)
    if att_df is not None and len(att_df) > 0:
        pe_att = att_df[att_df["outcome"] == "is_preeclampsia"]
        for treat in pe_att["treatment"].unique():
            sub = pe_att[pe_att["treatment"] == treat]
            records.append({
                "treatment": treat,
                "method": "PSW-ATT",
                "estimate": sub["att"].mean(),
                "ci_low": sub["att"].mean() - 1.96 * sub["att"].std() / np.sqrt(len(sub)),
                "ci_high": sub["att"].mean() + 1.96 * sub["att"].std() / np.sqrt(len(sub)),
            })

    # IV (全样本)
    if iv_df is not None and len(iv_df) > 0:
        pe_iv = iv_df[(iv_df["outcome"] == "is_preeclampsia") &
                       (iv_df["scope"] == "all") & (iv_df["status"] == "ok")]
        for _, r in pe_iv.iterrows():
            records.append({
                "treatment": r["treatment"],
                "method": "IV-2SLS",
                "estimate": r["coef"],
                "ci_low": r.get("ci_low", r["coef"] - 1.96 * r.get("se", 0)),
                "ci_high": r.get("ci_high", r["coef"] + 1.96 * r.get("se", 0)),
            })

    # RDD (全样本)
    if rdd_df is not None and len(rdd_df) > 0:
        pe_rdd = rdd_df[(rdd_df["outcome"] == "is_preeclampsia") &
                         (rdd_df["scope"] == "all") & (rdd_df["status"] == "ok")]
        for _, r in pe_rdd.iterrows():
            records.append({
                "treatment": r["treatment"],
                "method": "RDD",
                "estimate": r["tau"],
                "ci_low": r.get("ci_low", r["tau"] - 1.96 * r.get("se", 0)),
                "ci_high": r.get("ci_high", r["tau"] + 1.96 * r.get("se", 0)),
            })

    if not records:
        print("  无可绘图数据")
        return

    comp_df = pd.DataFrame(records)

    # 绘制 Forest Plot
    treatments = comp_df["treatment"].unique()
    methods = ["PSW-ATT", "IV-2SLS", "RDD"]
    method_colors = {"PSW-ATT": LANCET_COLORS["blue"],
                     "IV-2SLS": LANCET_COLORS["red"],
                     "RDD": LANCET_COLORS["green"]}

    fig, ax = plt.subplots(figsize=get_figure_size("single"))
    y_pos = 0
    y_ticks = []
    y_labels = []

    for treat in treatments:
        sub = comp_df[comp_df["treatment"] == treat]
        for method in methods:
            msub = sub[sub["method"] == method]
            if len(msub) == 0:
                continue
            r = msub.iloc[0]
            color = method_colors.get(method, "#999")

            ax.plot(r["estimate"], y_pos, "o", color=color, markersize=6, zorder=3)
            if not np.isnan(r["ci_low"]) and not np.isnan(r["ci_high"]):
                ax.plot([r["ci_low"], r["ci_high"]], [y_pos, y_pos],
                       "-", color=color, linewidth=1.5, zorder=2)
            y_ticks.append(y_pos)
            y_labels.append(f"{treat}\n({method})")
            y_pos -= 1
        y_pos -= 0.5  # 组间间隔

    ax.axvline(x=0, color="#999", linestyle="--", linewidth=0.5)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=6)
    ax.set_xlabel("Causal Effect Estimate", fontsize=9)
    ax.set_title("Robustness: ATT vs IV vs RDD\n(Preeclampsia)", fontsize=10, fontweight="bold")

    # 图例
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker="o", color=c, label=m, markersize=5, linestyle="-")
                       for m, c in method_colors.items()]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=7, frameon=False)

    plt.tight_layout()
    for fmt in ["pdf", "png"]:
        fig.savefig(out_dir / f"robustness_forest.{fmt}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  稳健性 Forest Plot 已保存")


def plot_robustness_summary_table(att_df, iv_df, rdd_df):
    """生成稳健性对比汇总表"""
    out_dir = OUTPUT_ROOT / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for treat_name in ["D_Dimer_high", "PLT_low", "ALT_high", "UA_high", "PRO_positive", "FIB_high"]:
        row = {"treatment": treat_name}

        # ATT
        if att_df is not None and len(att_df) > 0:
            pe = att_df[(att_df["treatment"] == treat_name) & (att_df["outcome"] == "is_preeclampsia")]
            row["att_mean"] = pe["att"].mean() if len(pe) > 0 else np.nan
            row["att_n_sig"] = int((pe["att"].abs() > 0.01).sum()) if len(pe) > 0 else 0

        # IV
        if iv_df is not None and len(iv_df) > 0:
            pe_iv = iv_df[(iv_df["treatment"] == treat_name) &
                           (iv_df["outcome"] == "is_preeclampsia") &
                           (iv_df["scope"] == "all") & (iv_df["status"] == "ok")]
            if len(pe_iv) > 0:
                row["iv_coef"] = pe_iv.iloc[0]["coef"]
                row["iv_p"] = pe_iv.iloc[0]["p_value"]
                row["iv_f1"] = pe_iv.iloc[0].get("first_stage_f", np.nan)
            else:
                row["iv_coef"] = np.nan

        # RDD
        if rdd_df is not None and len(rdd_df) > 0:
            pe_rdd = rdd_df[(rdd_df["treatment"] == treat_name) &
                             (rdd_df["outcome"] == "is_preeclampsia") &
                             (rdd_df["scope"] == "all") & (rdd_df["status"] == "ok")]
            if len(pe_rdd) > 0:
                row["rdd_tau"] = pe_rdd.iloc[0]["tau"]
                row["rdd_p"] = pe_rdd.iloc[0]["p_value"]
                row["rdd_density_p"] = pe_rdd.iloc[0].get("density_p", np.nan)
            else:
                row["rdd_tau"] = np.nan

        # 一致性判断: 三种方法符号一致
        signs = []
        for key in ["att_mean", "iv_coef", "rdd_tau"]:
            v = row.get(key, np.nan)
            if not np.isnan(v):
                signs.append(np.sign(v))
        row["sign_consistent"] = len(set(signs)) <= 1 if signs else False

        records.append(row)

    summary = pd.DataFrame(records)
    summary.to_csv(out_dir / "robustness_comparison.csv", index=False)
    print("\n  稳健性对比表已保存")
    return summary


def run(att_df=None, iv_df=None, rdd_df=None):
    plot_method_comparison(att_df, iv_df, rdd_df)
    summary = plot_robustness_summary_table(att_df, iv_df, rdd_df)

    print("\n=== 稳健性汇总 (PE) ===")
    if summary is not None and len(summary) > 0:
        cols = [c for c in ["treatment", "att_mean", "iv_coef", "iv_p", "iv_f1",
                            "rdd_tau", "rdd_p", "sign_consistent"] if c in summary.columns]
        print(summary[cols].to_string(index=False))
    return summary

"""
修复脚本: 运行 NOTEARS → 重新集成 4 方法 → 重跑级联/反事实/可视化/IV/RDD
前提: PC/FCI/LiNGAM 结果已存在
"""

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from config.settings import OUTPUT_ROOT


def main():
    t0 = time.time()
    print("=" * 60)
    print("修复运行: NOTEARS + 4方法集成 + 全流程更新")
    print("=" * 60)

    # 加载数据集
    datasets = _load_datasets()
    if not datasets:
        print("ERROR: 无数据集")
        return

    graph_dir = OUTPUT_ROOT / "causal_graphs"

    # ── Step 1: NOTEARS-MLP (GPU) ──
    print("\n" + "=" * 50)
    print("Step 1: NOTEARS-MLP (GPU)")
    print("=" * 50)

    from causal_discovery.notears_discovery import run as notears_run
    notears_results = notears_run(datasets, n_bootstrap=20, device="cuda")

    # ── Step 2: 加载已有 PC/FCI/LiNGAM + 集成 ──
    print("\n" + "=" * 50)
    print("Step 2: 4 方法集成 DAG")
    print("=" * 50)

    pc_fci_results = _load_pc_fci(graph_dir, datasets)
    lingam_results = _load_lingam(graph_dir, datasets)

    from causal_discovery.ensemble_dag import run as ensemble_run
    consensus_dags = ensemble_run(pc_fci_results, {}, notears_results, lingam_results, datasets)

    # ── Step 3: 跨阶段比较 ──
    print("\n" + "=" * 50)
    print("Step 3: 跨阶段比较")
    print("=" * 50)

    from causal_discovery.trimester_comparison import run as compare_run
    comparison = compare_run(consensus_dags)

    # ── Step 4: 因果级联 ──
    print("\n" + "=" * 50)
    print("Step 4: 因果级联分析")
    print("=" * 50)

    from cascade_analysis.cascade_identification import run as cascade_run
    all_cascades = cascade_run(consensus_dags)

    from cascade_analysis.pathway_scoring import run as score_run
    scored_cascades = score_run(all_cascades)

    from cascade_analysis.outcome_specific_cascades import run as disease_run
    disease_run(scored_cascades)

    # ── Step 5: 反事实 ──
    print("\n" + "=" * 50)
    print("Step 5: 反事实推断 (ATT 窗口)")
    print("=" * 50)

    scaled_all, raw_all = _load_harmonized()

    from data_prep.build_causal_dataset import build_outcome_binary, binarize_treatments
    scaled_all = build_outcome_binary(scaled_all)
    raw_all = build_outcome_binary(raw_all)
    raw_with_treat = binarize_treatments(raw_all)
    for tc in raw_with_treat.columns:
        if tc not in scaled_all.columns:
            scaled_all[tc] = raw_with_treat[tc].values[:len(scaled_all)]

    from counterfactual.window_quantification import run as window_run
    window_results = window_run(scaled_all, raw_all)

    from counterfactual.sensitivity_analysis import run as sens_run
    sens_run(window_results)

    # ── Step 6: IV + RDD ──
    print("\n" + "=" * 50)
    print("Step 6: 稳健性检验 (IV + RDD)")
    print("=" * 50)

    from counterfactual.iv_analysis import run as iv_run
    iv_df = iv_run(scaled_all, raw_all)

    from counterfactual.rdd_analysis import run as rdd_run
    rdd_df = rdd_run(raw_all)

    # ── Step 7: 临床验证 ──
    print("\n" + "=" * 50)
    print("Step 7: 临床验证")
    print("=" * 50)

    from validation.guideline_comparison import run as guideline_run
    guideline_run(window_results, scored_cascades,
                  comparison.get("edge_comparison") if comparison else None)

    from validation.cross_center_replication import run as repl_run
    repl_run(consensus_dags)

    # ── Step 8: 可视化 ──
    print("\n" + "=" * 50)
    print("Step 8: 可视化")
    print("=" * 50)

    from visualization.dag_plots import run as dag_viz_run
    dag_viz_run(consensus_dags)

    from visualization.window_heatmap import run as heatmap_run
    heatmap_run(window_results)

    from visualization.cascade_sankey import run as sankey_run
    sankey_run(scored_cascades)

    from visualization.trimester_comparison_fig import run as tri_viz_run
    if comparison:
        tri_viz_run(comparison.get("edge_comparison"), comparison.get("shd"))

    # 稳健性图
    att_path = OUTPUT_ROOT / "counterfactual" / "window_att_results.csv"
    att_df = pd.read_csv(att_path) if att_path.exists() else pd.DataFrame()
    from visualization.robustness_plots import run as rob_viz_run
    rob_viz_run(att_df, iv_df, rdd_df)

    # ── 完成 ──
    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    print(f"全部修复完成! 耗时: {elapsed/60:.1f} 分钟")
    print("=" * 60)

    # 产出统计
    for subdir in ["causal_graphs", "cascades", "counterfactual", "figures", "tables"]:
        d = OUTPUT_ROOT / subdir
        if d.exists():
            files = list(d.iterdir())
            print(f"  {subdir}/: {len(files)} files")


def _load_datasets():
    """加载因果数据集"""
    causal_dir = OUTPUT_ROOT / "causal_datasets"
    feat_df = pd.read_csv(causal_dir / "selected_features.csv") if (causal_dir / "selected_features.csv").exists() else None
    datasets = {}
    for tri in ["T1", "T2", "T3"]:
        raw_path = causal_dir / f"{tri}_raw.parquet"
        if not raw_path.exists():
            continue
        raw_df = pd.read_parquet(raw_path)
        imputed = [pd.read_parquet(causal_dir / f"{tri}_imputed_{i}.parquet")
                   for i in range(5) if (causal_dir / f"{tri}_imputed_{i}.parquet").exists()]
        if not imputed:
            continue
        features = feat_df[feat_df["trimester"] == tri]["feature"].tolist() if feat_df is not None else []
        datasets[tri] = {"features": features, "n_records": len(raw_df),
                         "imputed": imputed, "raw": raw_df}
        print(f"  {tri}: {len(raw_df)} records, {len(features)} features")
    return datasets


def _load_harmonized():
    """加载标准化数据"""
    harm_dir = OUTPUT_ROOT / "harmonized"
    scaled = pd.concat([pd.read_parquet(f) for f in harm_dir.glob("harmonized_*.parquet")], ignore_index=True)
    raw = pd.concat([pd.read_parquet(f) for f in harm_dir.glob("raw_*.parquet")], ignore_index=True)
    return scaled, raw


def _load_pc_fci(graph_dir, datasets):
    """加载 PC/FCI 频率矩阵"""
    results = {}
    for tri_name in datasets:
        tri_results = {}
        for group in ["all", "PE", "normal"]:
            group_results = {}
            for method in ["pc", "fci"]:
                fpath = graph_dir / f"{tri_name}_{group}_{method}_freq.csv"
                if fpath.exists():
                    freq_df = pd.read_csv(fpath, index_col=0)
                    group_results[method] = freq_df.values
            if group_results:
                tri_results[group] = group_results
        if tri_results:
            results[tri_name] = tri_results
    print(f"  PC/FCI: {list(results.keys())}")
    return results


def _load_lingam(graph_dir, datasets):
    """加载 LiNGAM 频率矩阵"""
    results = {}
    for tri_name in datasets:
        fpath = graph_dir / f"{tri_name}_all_lingam_freq.csv"
        if fpath.exists():
            freq_df = pd.read_csv(fpath, index_col=0)
            results[tri_name] = {"freq_matrix": freq_df.values, "features": datasets[tri_name]["features"]}
    print(f"  LiNGAM: {list(results.keys())}")
    return results


if __name__ == "__main__":
    main()

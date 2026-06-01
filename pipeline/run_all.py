"""
端到端流程编排:
Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5 → Phase 6
"""

import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from config.settings import OUTPUT_ROOT


def main():
    t0 = time.time()
    print("=" * 70)
    print("方向二: 妊娠并发症因果发现与可干预窗口期识别")
    print("=" * 70)

    # ── Phase 1: 数据准备 ──
    print("\n" + "=" * 50)
    print("Phase 1: 数据准备")
    print("=" * 50)

    harm_dir = OUTPUT_ROOT / "harmonized"
    if not (harm_dir / "harmonized_SFY.parquet").exists():
        print("运行数据标准化...")
        from data_prep.harmonize_local import run as harmonize_run
        harmonize_run()
    else:
        print("标准化数据已存在，跳过")

    causal_dir = OUTPUT_ROOT / "causal_datasets"
    if not (causal_dir / "T2_imputed_0.parquet").exists():
        print("\n构建因果数据集...")
        from data_prep.build_causal_dataset import run as build_run
        datasets = build_run()
    else:
        print("\n因果数据集已存在，加载...")
        datasets = _load_existing_datasets(causal_dir)

    if not datasets:
        print("ERROR: 无有效数据集")
        return

    # ── Phase 2: 因果发现 ──
    print("\n" + "=" * 50)
    print("Phase 2: 因果发现")
    print("=" * 50)

    # 降低 bootstrap 次数以加速首次运行（正式版可改为 500）
    N_BOOT = 50

    print("\n--- PC + FCI ---")
    graph_dir = OUTPUT_ROOT / "causal_graphs"
    pc_fci_results = _load_existing_pc_fci(graph_dir, datasets)
    if not pc_fci_results:
        from causal_discovery.pc_fci_discovery import run as pc_fci_run
        try:
            pc_fci_results = pc_fci_run(datasets, alpha=0.01, n_bootstrap=N_BOOT)
        except Exception as e:
            print(f"  PC/FCI 错误: {e}")
            pc_fci_results = {}
    else:
        print("  已加载已有 PC/FCI 结果")

    print("\n--- Granger ---")
    from causal_discovery.granger_population import run as granger_run
    try:
        granger_results = granger_run(datasets, lag_weeks=4)
    except Exception as e:
        print(f"  Granger 错误: {e}")
        granger_results = {}

    print("\n--- NOTEARS-MLP (GPU) ---")
    from causal_discovery.notears_discovery import run as notears_run
    try:
        notears_results = notears_run(datasets, n_bootstrap=20, device="cuda")
    except Exception as e:
        print(f"  NOTEARS 错误: {e}")
        notears_results = {}

    print("\n--- LiNGAM ---")
    from causal_discovery.lingam_discovery import run as lingam_run
    try:
        lingam_results = lingam_run(datasets, n_bootstrap=N_BOOT)
    except Exception as e:
        print(f"  LiNGAM 错误: {e}")
        lingam_results = {}

    print("\n--- 集成 DAG ---")
    from causal_discovery.ensemble_dag import run as ensemble_run
    consensus_dags = ensemble_run(pc_fci_results, granger_results,
                                   notears_results, lingam_results, datasets)

    print("\n--- 跨阶段比较 ---")
    from causal_discovery.trimester_comparison import run as compare_run
    comparison = compare_run(consensus_dags)

    # ── Phase 3: 因果级联分析 ──
    print("\n" + "=" * 50)
    print("Phase 3: 因果级联分析")
    print("=" * 50)

    from cascade_analysis.cascade_identification import run as cascade_run
    all_cascades = cascade_run(consensus_dags)

    from cascade_analysis.pathway_scoring import run as score_run
    scored_cascades = score_run(all_cascades)

    from cascade_analysis.outcome_specific_cascades import run as disease_run
    disease_cascades = disease_run(scored_cascades)

    # ── Phase 4: 反事实推断 ──
    print("\n" + "=" * 50)
    print("Phase 4: 反事实推断与可干预窗口")
    print("=" * 50)

    # 加载合并的标准化数据用于窗口分析
    import pandas as pd
    scaled_frames = []
    raw_frames = []
    for center_file in (OUTPUT_ROOT / "harmonized").glob("harmonized_*.parquet"):
        scaled_frames.append(pd.read_parquet(center_file))
    for center_file in (OUTPUT_ROOT / "harmonized").glob("raw_*.parquet"):
        raw_frames.append(pd.read_parquet(center_file))
    scaled_all = pd.concat(scaled_frames, ignore_index=True) if scaled_frames else pd.DataFrame()
    raw_all = pd.concat(raw_frames, ignore_index=True) if raw_frames else pd.DataFrame()

    # 添加二值变量
    from data_prep.build_causal_dataset import build_outcome_binary, binarize_treatments
    scaled_all = build_outcome_binary(scaled_all)
    raw_with_treat = binarize_treatments(raw_all)
    treat_cols = list(raw_with_treat.columns)
    for tc in treat_cols:
        if tc not in scaled_all.columns and tc in raw_with_treat.columns:
            scaled_all[tc] = raw_with_treat[tc].values[:len(scaled_all)] if len(raw_with_treat) >= len(scaled_all) else None

    from counterfactual.window_quantification import run as window_run
    window_results = window_run(scaled_all, raw_all)

    from counterfactual.sensitivity_analysis import run as sens_run
    sens_results = sens_run(window_results)

    # ── Phase 5: 临床验证 ──
    print("\n" + "=" * 50)
    print("Phase 5: 临床验证")
    print("=" * 50)

    from validation.guideline_comparison import run as guideline_run
    guideline_results = guideline_run(window_results, scored_cascades,
                                       comparison.get("edge_comparison") if comparison else None)

    from validation.cross_center_replication import run as repl_run
    repl_results = repl_run(consensus_dags)

    # ── Phase 6: 可视化 ──
    print("\n" + "=" * 50)
    print("Phase 6: 可视化")
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

    # ── 完成 ──
    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print(f"全部完成! 耗时: {elapsed/60:.1f} 分钟")
    print(f"结果输出到: {OUTPUT_ROOT}")
    print("=" * 70)

    # 列出产出文件
    print("\n产出文件:")
    for subdir in ["harmonized", "causal_datasets", "causal_graphs", "cascades",
                    "counterfactual", "figures", "tables"]:
        d = OUTPUT_ROOT / subdir
        if d.exists():
            files = list(d.iterdir())
            print(f"  {subdir}/: {len(files)} files")
            for f in sorted(files)[:5]:
                print(f"    {f.name}")
            if len(files) > 5:
                print(f"    ... ({len(files)-5} more)")


def _load_existing_pc_fci(graph_dir: Path, datasets: dict) -> dict:
    """加载已有的 PC/FCI 频率矩阵"""
    import pandas as pd
    import numpy as np
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
    if results:
        print(f"  加载已有 PC/FCI: {list(results.keys())}")
    return results


def _load_existing_datasets(causal_dir: Path) -> dict:
    """加载已有的因果数据集"""
    import pandas as pd
    datasets = {}
    feat_df = pd.read_csv(causal_dir / "selected_features.csv") if (causal_dir / "selected_features.csv").exists() else None

    for tri in ["T1", "T2", "T3"]:
        raw_path = causal_dir / f"{tri}_raw.parquet"
        if not raw_path.exists():
            continue

        raw_df = pd.read_parquet(raw_path)
        imputed = []
        for i in range(5):
            imp_path = causal_dir / f"{tri}_imputed_{i}.parquet"
            if imp_path.exists():
                imputed.append(pd.read_parquet(imp_path))

        if not imputed:
            continue

        features = feat_df[feat_df["trimester"] == tri]["feature"].tolist() if feat_df is not None else []
        if not features:
            # 推断
            import numpy as np
            from config.settings import ALL_FEATURES
            features = [f for f in ALL_FEATURES if f in raw_df.columns and raw_df[f].notna().mean() > 0.3]

        datasets[tri] = {
            "features": features,
            "n_records": len(raw_df),
            "imputed": imputed,
            "raw": raw_df,
        }
        print(f"  加载 {tri}: {len(raw_df)} records, {len(features)} features, {len(imputed)} imputed sets")

    return datasets


if __name__ == "__main__":
    main()

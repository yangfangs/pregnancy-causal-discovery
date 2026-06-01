"""
单独运行稳健性检验: IV + RDD + 可视化对比
依赖已有的 harmonized 数据和 window_att_results
"""

import sys
import time
import warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from config.settings import OUTPUT_ROOT


def main():
    t0 = time.time()
    print("=" * 60)
    print("稳健性检验: 工具变量 (IV) + 断点回归 (RDD)")
    print("=" * 60)

    # 加载数据
    harm_dir = OUTPUT_ROOT / "harmonized"
    raw_frames = []
    scaled_frames = []
    for f in harm_dir.glob("raw_*.parquet"):
        raw_frames.append(pd.read_parquet(f))
    for f in harm_dir.glob("harmonized_*.parquet"):
        scaled_frames.append(pd.read_parquet(f))
    raw_all = pd.concat(raw_frames, ignore_index=True)
    scaled_all = pd.concat(scaled_frames, ignore_index=True)
    print(f"数据: {len(raw_all)} records (raw), {len(scaled_all)} records (scaled)")

    # 加载已有 ATT 结果
    att_path = OUTPUT_ROOT / "counterfactual" / "window_att_results.csv"
    att_df = pd.read_csv(att_path) if att_path.exists() else pd.DataFrame()

    # ── IV 分析 ──
    print("\n" + "=" * 50)
    print("工具变量 (IV-2SLS) 分析")
    print("=" * 50)

    from counterfactual.iv_analysis import run as iv_run
    iv_df = iv_run(scaled_all, raw_all)

    # ── RDD 分析 ──
    print("\n" + "=" * 50)
    print("断点回归 (RDD) 分析")
    print("=" * 50)

    from counterfactual.rdd_analysis import run as rdd_run
    rdd_df = rdd_run(raw_all)

    # ── 可视化 ──
    print("\n" + "=" * 50)
    print("稳健性可视化")
    print("=" * 50)

    from visualization.robustness_plots import run as viz_run
    summary = viz_run(att_df, iv_df, rdd_df)

    elapsed = time.time() - t0
    print(f"\n完成! 耗时: {elapsed/60:.1f} 分钟")
    print(f"产出:")
    print(f"  {OUTPUT_ROOT / 'counterfactual' / 'iv_2sls_results.csv'}")
    print(f"  {OUTPUT_ROOT / 'counterfactual' / 'rdd_results.csv'}")
    print(f"  {OUTPUT_ROOT / 'figures' / 'robustness_forest.pdf'}")
    print(f"  {OUTPUT_ROOT / 'tables' / 'robustness_comparison.csv'}")


if __name__ == "__main__":
    main()

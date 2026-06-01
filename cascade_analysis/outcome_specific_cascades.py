"""
疾病特异性级联对比: PE vs GDM vs HTN
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import OUTPUT_ROOT


def compare_disease_cascades(scored_cascades: dict) -> pd.DataFrame:
    """
    跨 trimester 汇总，标记高分路径的器官系统模式
    """
    out_dir = OUTPUT_ROOT / "cascades"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_paths = []
    for tri_name, df in scored_cascades.items():
        if df is None or len(df) == 0:
            continue
        top = df.head(20).copy()
        top["trimester"] = tri_name
        all_paths.append(top)

    if not all_paths:
        return pd.DataFrame()

    combined = pd.concat(all_paths, ignore_index=True)

    # 根据终止系统分类，哪些级联可能与特定疾病相关
    # 终止于 urine (U_PRO) → PE相关; 终止于 metabolic (C_Glu) → GDM相关
    combined["likely_disease"] = combined["end_system"].map({
        "urine": "PE",
        "renal": "PE/HTN",
        "metabolic": "GDM",
        "coagulation": "PE/HTN",
        "liver": "PE (HELLP)",
    }).fillna("general")

    combined.to_csv(out_dir / "disease_specific_cascades.csv", index=False)

    # 统计
    print("\n=== 疾病特异性级联分布 ===")
    print(combined.groupby(["trimester", "likely_disease"]).size().unstack(fill_value=0))

    return combined


def run(scored_cascades: dict):
    return compare_disease_cascades(scored_cascades)

"""
构建因果分析数据集：
1. 加载标准化 + 原始数据
2. 按孕期三阶段分层 (T1/T2/T3)
3. 特征选择（覆盖率 + 方差 + 共线性）
4. MICE 多重插补 (M=5)
5. 二值化治疗变量
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from scipy.stats import iqr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import (
    ALL_FEATURES, CENTERS, GW_WINDOWS, OUTPUT_ROOT, TREATMENT_THRESHOLDS,
    TRIMESTER_BOUNDS, MIN_FEATURE_COVERAGE, MIN_STRATUM_SIZE,
)
from data_prep.feature_selection import select_features

warnings.filterwarnings("ignore")


def load_harmonized():
    """加载所有中心的标准化数据和原始数据"""
    harm_dir = OUTPUT_ROOT / "harmonized"
    scaled_frames, raw_frames = [], []
    for center in CENTERS:
        sp = harm_dir / f"harmonized_{center}.parquet"
        rp = harm_dir / f"raw_{center}.parquet"
        if sp.exists():
            scaled_frames.append(pd.read_parquet(sp))
        if rp.exists():
            raw_frames.append(pd.read_parquet(rp))
    scaled = pd.concat(scaled_frames, ignore_index=True) if scaled_frames else pd.DataFrame()
    raw = pd.concat(raw_frames, ignore_index=True) if raw_frames else pd.DataFrame()
    return scaled, raw


def stratify_by_trimester(df: pd.DataFrame) -> dict:
    """按孕期分层，排除 GW=-1"""
    df = df[df["gestational_week"] > 0].copy()
    strata = {}
    for name, (lo, hi) in TRIMESTER_BOUNDS.items():
        mask = (df["gestational_week"] >= lo) & (df["gestational_week"] < hi)
        sub = df[mask].copy()
        if len(sub) >= MIN_STRATUM_SIZE:
            strata[name] = sub
            print(f"  {name}: {len(sub)} records, GW [{lo}, {hi})")
        else:
            print(f"  {name}: {len(sub)} records (跳过, < {MIN_STRATUM_SIZE})")
    return strata


def winsorize(df: pd.DataFrame, features: list, lower=0.01, upper=0.99) -> pd.DataFrame:
    """Winsorize 异常值到 1-99 百分位"""
    df = df.copy()
    for col in features:
        if col in df.columns:
            lo_val = df[col].quantile(lower)
            hi_val = df[col].quantile(upper)
            df[col] = df[col].clip(lo_val, hi_val)
    return df


def mice_impute(df: pd.DataFrame, features: list, m: int = 5, max_iter: int = 10) -> list:
    """MICE 多重插补，返回 M 个插补后的 DataFrame"""
    X = df[features].values
    imputed_list = []
    for i in range(m):
        imp = IterativeImputer(max_iter=max_iter, random_state=42 + i, sample_posterior=True)
        X_imp = imp.fit_transform(X)
        df_imp = df.copy()
        df_imp[features] = X_imp
        imputed_list.append(df_imp)
        print(f"    插补 {i+1}/{m} 完成")
    return imputed_list


def binarize_treatments(raw_df: pd.DataFrame) -> pd.DataFrame:
    """基于临床阈值二值化治疗变量（使用原始值）"""
    df = raw_df.copy()
    for treat_name, (feat, op, thresh) in TREATMENT_THRESHOLDS.items():
        if feat not in df.columns:
            df[treat_name] = np.nan
            continue
        if op == ">":
            df[treat_name] = (df[feat] > thresh).astype(float)
        elif op == "<":
            df[treat_name] = (df[feat] < thresh).astype(float)
        elif op == ">=":
            df[treat_name] = (df[feat] >= thresh).astype(float)
        else:
            df[treat_name] = np.nan
        df.loc[df[feat].isna(), treat_name] = np.nan
    return df


def build_outcome_binary(df: pd.DataFrame) -> pd.DataFrame:
    """添加二值结局列"""
    df = df.copy()
    df["is_preeclampsia"] = (df["outcome"] == "preeclampsia").astype(int)
    df["is_gdm"] = (df["outcome"] == "gdm").astype(int)
    df["is_hypertension"] = (df["outcome"] == "hypertension").astype(int)
    df["is_adverse"] = df[["is_preeclampsia", "is_gdm", "is_hypertension"]].max(axis=1)
    return df


def run():
    out_dir = OUTPUT_ROOT / "causal_datasets"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("加载标准化数据...")
    scaled_all, raw_all = load_harmonized()
    print(f"  标准化: {len(scaled_all)} records, 原始: {len(raw_all)} records")

    # 添加二值结局
    scaled_all = build_outcome_binary(scaled_all)
    raw_all = build_outcome_binary(raw_all)

    # 二值化治疗变量（基于原始值）
    raw_with_treat = binarize_treatments(raw_all)
    treat_cols = list(TREATMENT_THRESHOLDS.keys())

    # 把治疗变量合并到标准化数据
    if len(scaled_all) == len(raw_with_treat):
        for tc in treat_cols:
            scaled_all[tc] = raw_with_treat[tc].values

    print("\n按孕期分层...")
    strata = stratify_by_trimester(scaled_all)

    results = {}
    for tri_name, tri_df in strata.items():
        print(f"\n=== {tri_name} ===")

        # 特征选择
        selected_feats = select_features(tri_df, ALL_FEATURES)
        print(f"  选定 {len(selected_feats)} 个特征: {selected_feats}")

        if len(selected_feats) < 5:
            print(f"  特征不足, 跳过 {tri_name}")
            continue

        # Winsorize
        tri_df = winsorize(tri_df, selected_feats)

        # MICE 多重插补
        print(f"  MICE 多重插补 (M=5)...")
        imputed_list = mice_impute(tri_df, selected_feats, m=5)

        # 保存
        for i, imp_df in enumerate(imputed_list):
            imp_df.to_parquet(out_dir / f"{tri_name}_imputed_{i}.parquet", index=False)

        # 同时保存不插补的完整版（带 NaN）
        tri_df.to_parquet(out_dir / f"{tri_name}_raw.parquet", index=False)

        results[tri_name] = {
            "features": selected_feats,
            "n_records": len(tri_df),
            "imputed": imputed_list,
            "raw": tri_df,
        }

    # 保存特征选择结果
    feat_summary = []
    for tri_name, info in results.items():
        for f in info["features"]:
            feat_summary.append({"trimester": tri_name, "feature": f})
    pd.DataFrame(feat_summary).to_csv(out_dir / "selected_features.csv", index=False)

    print(f"\n因果数据集构建完成! 输出到 {out_dir}")
    return results


if __name__ == "__main__":
    run()

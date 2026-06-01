"""
群体级面板 Granger 因果检验
由于个体时间序列稀疏，使用截面数据模拟时间效应:
对于 (X, Y) 对，检验 X 在 GW=t-delta 时的值是否预测 Y 在 GW=t 时的值
"""

import sys
import itertools
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import OUTPUT_ROOT

warnings.filterwarnings("ignore")


def panel_granger_test(df: pd.DataFrame, cause: str, effect: str,
                        lag_weeks: int = 4, min_pairs: int = 100):
    """
    群体级 Granger 因果检验:
    将数据按 patient_id 连接不同孕周的检验值，
    检验 cause_t-lag 是否预测 effect_t (控制 effect_t-lag)
    """
    df = df[["patient_id", "gestational_week", cause, effect]].dropna()

    # 构建 (patient, gw) → (patient, gw+lag) 配对
    df_early = df.rename(columns={cause: f"{cause}_lag", effect: f"{effect}_lag",
                                   "gestational_week": "gw_early"})
    df_late = df.rename(columns={cause: f"{cause}_now", effect: f"{effect}_now",
                                  "gestational_week": "gw_late"})

    merged = df_early.merge(df_late, on="patient_id")
    merged["gw_diff"] = merged["gw_late"] - merged["gw_early"]
    # 允许 lag ± 2 周的容差
    paired = merged[(merged["gw_diff"] >= lag_weeks - 2) & (merged["gw_diff"] <= lag_weeks + 2)]

    if len(paired) < min_pairs:
        return None

    # 偏相关: cause_lag → effect_now | effect_lag
    from scipy.stats import pearsonr
    from numpy.linalg import lstsq

    # 残差化: 回归 effect_now ~ effect_lag, 取残差
    X_control = paired[f"{effect}_lag"].values.reshape(-1, 1)
    X_control = np.column_stack([X_control, np.ones(len(X_control))])
    y = paired[f"{effect}_now"].values

    coef, _, _, _ = lstsq(X_control, y, rcond=None)
    resid_y = y - X_control @ coef

    # 残差化: 回归 cause_lag ~ effect_lag, 取残差
    x_cause = paired[f"{cause}_lag"].values
    coef2, _, _, _ = lstsq(X_control, x_cause, rcond=None)
    resid_x = x_cause - X_control @ coef2

    # 偏相关
    if np.std(resid_x) < 1e-10 or np.std(resid_y) < 1e-10:
        return None

    r, p = pearsonr(resid_x, resid_y)

    return {
        "cause": cause,
        "effect": effect,
        "partial_r": r,
        "p_value": p,
        "n_pairs": len(paired),
        "lag_weeks": lag_weeks,
    }


def run(datasets: dict, lag_weeks: int = 4):
    """对每个 trimester 运行所有特征对的 Granger 检验"""
    out_dir = OUTPUT_ROOT / "causal_graphs"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for tri_name, tri_info in datasets.items():
        features = tri_info["features"]
        # 使用原始（未插补）数据以保留自然变异, 大数据集子采样
        df = tri_info["raw"].copy()
        if len(df) > 20000:
            df = df.sample(n=20000, random_state=42)

        print(f"\n=== Granger {tri_name} ({len(features)} features, n={len(df)}) ===")

        results = []
        pairs = list(itertools.permutations(features, 2))
        for i, (cause, effect) in enumerate(pairs):
            res = panel_granger_test(df, cause, effect, lag_weeks=lag_weeks)
            if res is not None:
                results.append(res)
            if (i + 1) % 100 == 0:
                print(f"    {i+1}/{len(pairs)} pairs tested")

        if not results:
            print(f"  {tri_name}: 无有效 Granger 结果")
            continue

        res_df = pd.DataFrame(results)

        # BH-FDR 校正
        _, pvals_corrected, _, _ = multipletests(res_df["p_value"], method="fdr_bh")
        res_df["p_fdr"] = pvals_corrected
        res_df["significant"] = res_df["p_fdr"] < 0.05

        # 转为频率矩阵格式（与 PC/FCI 对齐）
        n_feat = len(features)
        freq_matrix = np.zeros((n_feat, n_feat))
        feat_idx = {f: i for i, f in enumerate(features)}

        sig_df = res_df[res_df["significant"]]
        for _, row in sig_df.iterrows():
            i, j = feat_idx[row["cause"]], feat_idx[row["effect"]]
            freq_matrix[i, j] = abs(row["partial_r"])

        # 保存
        res_df.to_csv(out_dir / f"{tri_name}_granger_results.csv", index=False)
        freq_df = pd.DataFrame(freq_matrix, index=features, columns=features)
        freq_df.to_csv(out_dir / f"{tri_name}_all_granger_freq.csv")

        all_results[tri_name] = {"results": res_df, "freq_matrix": freq_matrix, "features": features}
        print(f"  {tri_name}: {len(sig_df)}/{len(res_df)} significant edges (FDR<0.05)")

    return all_results


if __name__ == "__main__":
    print("请通过 pipeline/run_all.py 运行")

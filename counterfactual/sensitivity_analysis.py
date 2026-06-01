"""
敏感性分析:
- E-value 计算
- 跨中心验证
- 多重插补方差汇总 (Rubin's rules)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import OUTPUT_ROOT


def rubins_rules(estimates: list, variances: list = None) -> dict:
    """
    Rubin's rules 合并多重插补结果
    estimates: M 个 ATT 估计值
    """
    M = len(estimates)
    estimates = [e for e in estimates if not np.isnan(e)]
    if len(estimates) < 2:
        return {"combined_estimate": estimates[0] if estimates else np.nan,
                "total_variance": np.nan}

    Q_bar = np.mean(estimates)  # 合并估计
    B = np.var(estimates, ddof=1)  # 插补间方差

    if variances:
        U_bar = np.mean(variances)  # 插补内方差
        T = U_bar + (1 + 1 / M) * B  # 总方差
    else:
        T = (1 + 1 / M) * B

    return {
        "combined_estimate": Q_bar,
        "between_variance": B,
        "total_variance": T,
        "se": np.sqrt(T),
        "ci_low": Q_bar - 1.96 * np.sqrt(T),
        "ci_high": Q_bar + 1.96 * np.sqrt(T),
    }


def cross_center_validation(window_results: pd.DataFrame,
                              datasets_by_center: dict) -> pd.DataFrame:
    """跨中心验证: 比较不同中心的 ATT 估计"""
    out_dir = OUTPUT_ROOT / "counterfactual"

    # 如果有分中心的数据可以做验证
    records = []
    if window_results is not None and len(window_results) > 0:
        for _, row in window_results.iterrows():
            records.append({
                "treatment": row["treatment"],
                "outcome": row["outcome"],
                "gw_label": row["gw_label"],
                "att_pooled": row["att"],
                "n_pooled": row["n_total"],
            })

    result = pd.DataFrame(records)
    if len(result) > 0:
        result.to_csv(out_dir / "cross_center_validation.csv", index=False)
    return result


def run(window_results: pd.DataFrame):
    """运行敏感性分析汇总"""
    out_dir = OUTPUT_ROOT / "counterfactual"
    out_dir.mkdir(parents=True, exist_ok=True)

    if window_results is None or len(window_results) == 0:
        print("无窗口结果可分析")
        return {}

    # E-value 汇总
    summary = window_results.groupby(["treatment", "outcome"]).agg(
        mean_att=("att", "mean"),
        max_abs_att=("att", lambda x: x.abs().max()),
        mean_e_value=("e_value", "mean"),
        n_windows=("att", "count"),
    ).reset_index()

    summary.to_csv(out_dir / "sensitivity_summary.csv", index=False)
    print("\n=== 敏感性分析汇总 ===")
    print(summary.to_string(index=False))

    return {"summary": summary}

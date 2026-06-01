"""
可干预窗口量化:
滑动孕周窗口 ATT 估计，识别峰值干预窗口
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import GW_WINDOWS, TREATMENT_THRESHOLDS, OUTPUT_ROOT
from counterfactual.dowhy_estimation import estimate_att, compute_e_value

warnings.filterwarnings("ignore")


def run(scaled_all: pd.DataFrame, raw_all: pd.DataFrame):
    """
    对每个治疗变量在每个孕周窗口估计 ATT
    scaled_all: 合并后的标准化数据（含 treatment 和 outcome 二值列）
    """
    out_dir = OUTPUT_ROOT / "counterfactual"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for treat_name, (feat, op, thresh) in TREATMENT_THRESHOLDS.items():
        if treat_name not in scaled_all.columns:
            continue

        for gw_lo, gw_hi in GW_WINDOWS:
            window_df = scaled_all[
                (scaled_all["gestational_week"] >= gw_lo)
                & (scaled_all["gestational_week"] < gw_hi)
            ].copy()

            if len(window_df) < 100:
                continue

            n_treated = int(window_df[treat_name].sum()) if treat_name in window_df.columns else 0
            n_control = len(window_df) - n_treated
            if n_treated < 20 or n_control < 20:
                continue

            # 混杂因素: age + 基线血常规/生化指标（排除当前 treatment 对应特征）
            base_confounders = ["age", "B_WBC", "B_HGB", "B_PLT", "B_RBC",
                                "C_ALT", "C_ALB", "C_Cr", "C_UA", "C_Glu"]
            treat_feat = TREATMENT_THRESHOLDS[treat_name][0]
            confounders = [c for c in base_confounders
                          if c in window_df.columns and c != treat_feat
                          and window_df[c].notna().mean() > 0.3]

            for outcome_col in ["is_preeclampsia", "is_hypertension", "is_gdm"]:
                if outcome_col not in window_df.columns:
                    continue

                # 检查结局变异
                if window_df[outcome_col].nunique() < 2:
                    continue

                try:
                    res = estimate_att(window_df, treat_name, outcome_col, confounders)
                    att = res["att"]
                    nnt = 1.0 / abs(att) if att and not np.isnan(att) and abs(att) > 1e-6 else np.nan

                    results.append({
                        "treatment": treat_name,
                        "outcome": outcome_col,
                        "gw_start": gw_lo,
                        "gw_end": gw_hi,
                        "gw_label": f"{gw_lo}-{gw_hi}w",
                        "att": att,
                        "nnt": nnt,
                        "e_value": compute_e_value(att),
                        "n_total": len(window_df),
                        "n_treated": n_treated,
                        "n_control": n_control,
                    })
                except Exception as e:
                    print(f"    {treat_name} → {outcome_col} @ {gw_lo}-{gw_hi}w: {e}")

    result_df = pd.DataFrame(results)
    if len(result_df) > 0:
        result_df.to_csv(out_dir / "window_att_results.csv", index=False)

        # 识别峰值窗口
        for treat in result_df["treatment"].unique():
            for outcome in result_df["outcome"].unique():
                sub = result_df[(result_df["treatment"] == treat) & (result_df["outcome"] == outcome)]
                if len(sub) == 0:
                    continue
                peak_idx = sub["att"].abs().idxmax()
                peak = sub.loc[peak_idx]
                print(f"  峰值: {treat} → {outcome} @ {peak['gw_label']} (ATT={peak['att']:.4f}, NNT={peak['nnt']:.1f})")

    print(f"\n窗口量化完成: {len(result_df)} 条记录")
    return result_df

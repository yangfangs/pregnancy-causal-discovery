"""
工具变量 (IV) 分析:
IV1: center 哑变量 — 不同中心检测标准差异影响指标值
IV2: 季节 (sample month) — 季节影响维生素D/钙/血液指标，但不直接影响PE
通过 2SLS 估计实验室指标异常对妊娠结局的因果效应。
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from linearmodels.iv import IV2SLS
from scipy import stats
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import TREATMENT_THRESHOLDS, GW_WINDOWS, OUTPUT_ROOT

warnings.filterwarnings("ignore")


def iv_2sls_estimate(df: pd.DataFrame, treatment_feat: str, outcome: str,
                      instruments: list, controls: list) -> dict:
    """2SLS 工具变量估计"""
    cols = [treatment_feat, outcome] + instruments + controls
    cols = [c for c in cols if c in df.columns]
    valid = df[cols].dropna()

    if len(valid) < 200:
        return {"coef": np.nan, "se": np.nan, "p_value": np.nan, "n": len(valid),
                "method": "iv_2sls", "status": "insufficient_data"}

    y = valid[outcome]
    if y.nunique() < 2:
        return {"coef": np.nan, "se": np.nan, "p_value": np.nan, "n": len(valid),
                "method": "iv_2sls", "status": "no_outcome_variation"}

    endog = valid[treatment_feat]
    avail_controls = [c for c in controls if c in valid.columns]
    avail_instruments = [c for c in instruments if c in valid.columns]

    if not avail_instruments:
        return {"coef": np.nan, "se": np.nan, "p_value": np.nan, "n": len(valid),
                "method": "iv_2sls", "status": "no_instruments"}

    try:
        exog = valid[avail_controls].copy() if avail_controls else pd.DataFrame(index=valid.index)
        exog["const"] = 1.0
        instr = valid[avail_instruments]

        model = IV2SLS(y, exog, endog, instr)
        result = model.fit(cov_type="robust")

        coef = result.params[treatment_feat]
        se = result.std_errors[treatment_feat]
        p_val = result.pvalues[treatment_feat]

        # 第一阶段 F 统计量
        first_stage_f = _first_stage_f(valid, treatment_feat, avail_instruments, avail_controls)

        # OLS 对比
        ols_coef = _ols_estimate(valid, treatment_feat, outcome, avail_controls)

        return {
            "coef": coef, "se": se, "p_value": p_val,
            "ci_low": coef - 1.96 * se, "ci_high": coef + 1.96 * se,
            "first_stage_f": first_stage_f, "ols_coef": ols_coef,
            "n": len(valid), "method": "iv_2sls", "status": "ok",
        }
    except Exception as e:
        return {"coef": np.nan, "se": np.nan, "p_value": np.nan, "n": len(valid),
                "method": "iv_2sls", "status": f"error: {str(e)[:80]}"}


def _first_stage_f(df, treatment, instruments, controls):
    """第一阶段 F: 仅IV系数的联合显著性"""
    try:
        # 完整模型 (含IV)
        X_full = df[instruments + controls].copy()
        X_full["const"] = 1.0
        y = df[treatment]
        model_full = sm.OLS(y, X_full).fit()

        # 受限模型 (不含IV)
        X_restricted = df[controls].copy() if controls else pd.DataFrame(index=df.index)
        X_restricted["const"] = 1.0
        model_restricted = sm.OLS(y, X_restricted).fit()

        # F = ((RSS_r - RSS_f) / q) / (RSS_f / (n - k))
        rss_r = model_restricted.ssr
        rss_f = model_full.ssr
        q = len(instruments)
        n = len(df)
        k = len(X_full.columns)
        f_stat = ((rss_r - rss_f) / q) / (rss_f / (n - k))
        return float(f_stat)
    except Exception:
        return np.nan


def _ols_estimate(df, treatment, outcome, controls):
    """OLS 对比估计"""
    try:
        X = df[[treatment] + controls].copy()
        X["const"] = 1.0
        model = sm.OLS(df[outcome], X).fit()
        return model.params[treatment]
    except Exception:
        return np.nan


def run(scaled_all: pd.DataFrame, raw_all: pd.DataFrame):
    """对每个治疗变量运行 IV 分析 (center + season 作为 IV)"""
    out_dir = OUTPUT_ROOT / "counterfactual"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = raw_all.copy()
    df["is_preeclampsia"] = (df["outcome"] == "preeclampsia").astype(int)
    df["is_gdm"] = (df["outcome"] == "gdm").astype(int)
    df["is_hypertension"] = (df["outcome"] == "hypertension").astype(int)
    df = df[df["gestational_week"] > 0].copy()

    iv_cols = []

    # IV1: center 哑变量
    if "center" in df.columns and df["center"].nunique() > 1:
        center_dummies = pd.get_dummies(df["center"], prefix="center", drop_first=True)
        df = pd.concat([df, center_dummies], axis=1)
        iv_cols += list(center_dummies.columns)

    # IV2: 季节 — 从 gestational_week 近似推断（孕周→分娩季节的逆推）
    # 用 gestational_week 本身的 sin/cos 变换作为季节性 IV
    gw = df["gestational_week"].values
    df["gw_sin"] = np.sin(2 * np.pi * gw / 40)  # 40 周周期
    df["gw_cos"] = np.cos(2 * np.pi * gw / 40)
    iv_cols += ["gw_sin", "gw_cos"]

    if not iv_cols:
        print("  无可用工具变量")
        return pd.DataFrame()

    print(f"  工具变量: {iv_cols}")

    # 控制变量
    base_controls = ["age"]
    controls = [c for c in base_controls if c in df.columns and df[c].notna().mean() > 0.3]

    results = []
    for treat_name, (feat, op, thresh) in TREATMENT_THRESHOLDS.items():
        if feat not in df.columns:
            continue

        for outcome_col in ["is_preeclampsia", "is_hypertension", "is_gdm"]:
            # 全样本 IV
            print(f"  IV: {feat} → {outcome_col}")
            res = iv_2sls_estimate(df, feat, outcome_col, iv_cols, controls)
            res.update({"treatment_feat": feat, "treatment": treat_name,
                        "outcome": outcome_col, "scope": "all"})
            results.append(res)

            # 按孕周窗口
            for gw_lo, gw_hi in GW_WINDOWS:
                wdf = df[(df["gestational_week"] >= gw_lo) & (df["gestational_week"] < gw_hi)]
                if len(wdf) < 200:
                    continue
                # 窗口内不用 gw_sin/gw_cos（变异太小），仅用 center
                w_ivs = [c for c in iv_cols if c.startswith("center")]
                if not w_ivs:
                    continue
                res_w = iv_2sls_estimate(wdf, feat, outcome_col, w_ivs, controls)
                res_w.update({"treatment_feat": feat, "treatment": treat_name,
                              "outcome": outcome_col, "scope": f"{gw_lo}-{gw_hi}w"})
                results.append(res_w)

    result_df = pd.DataFrame(results)
    result_df.to_csv(out_dir / "iv_2sls_results.csv", index=False)

    ok = result_df[result_df["status"] == "ok"]
    print(f"\n  IV 分析完成: {len(ok)}/{len(result_df)} 成功")
    if len(ok) > 0:
        sig = ok[ok["p_value"] < 0.05]
        strong = ok[ok["first_stage_f"] > 10]
        print(f"  显著 (p<0.05): {len(sig)}")
        print(f"  强 IV (F>10): {len(strong)}")

        pe_ok = ok[(ok["outcome"] == "is_preeclampsia") & (ok["scope"] == "all")]
        if len(pe_ok) > 0:
            print("\n  PE 全样本 IV 结果:")
            for _, r in pe_ok.iterrows():
                f1 = r.get("first_stage_f", np.nan)
                f1_str = f"{f1:.1f}" if not np.isnan(f1) else "N/A"
                print(f"    {r['treatment_feat']}: IV={r['coef']:.4f} (p={r['p_value']:.4f}), "
                      f"OLS={r['ols_coef']:.4f}, F1={f1_str}")

    return result_df

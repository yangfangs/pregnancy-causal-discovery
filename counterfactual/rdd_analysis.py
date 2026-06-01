"""
断点回归 (RDD) 分析:
利用临床诊断阈值作为"准实验"断点，估计指标跨越阈值时结局的跳变。

核心思想: 在临床阈值附近，恰好高于 vs 低于阈值的患者在其他特征上近似随机，
因此阈值处结局的跳变可解释为因果效应。

断点设置:
- D-Dimer: 0.5 mg/L FEU (DIC 诊断阈值)
- PLT: 150 × 10^9/L (血小板减少阈值)
- ALT: 40 U/L (肝功异常阈值)
- UA: 360 μmol/L (高尿酸阈值)
- Proteinuria: 0.5 (+/- 边界)
- FIB: 4.0 g/L (纤维蛋白原升高阈值)
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from rdrobust import rdrobust

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import TREATMENT_THRESHOLDS, GW_WINDOWS, OUTPUT_ROOT

warnings.filterwarnings("ignore")


def rdd_estimate(df: pd.DataFrame, running_var: str, outcome: str,
                  cutoff: float, bandwidth: float = None) -> dict:
    """
    Sharp RDD: 在 cutoff 处估计结局跳变
    running_var: 运行变量 (连续检验值)
    outcome: 二值结局
    cutoff: 临床阈值
    """
    valid = df[[running_var, outcome]].dropna()
    if len(valid) < 100:
        return {"tau": np.nan, "se": np.nan, "p_value": np.nan,
                "n": len(valid), "status": "insufficient_data"}

    y = valid[outcome].values
    x = valid[running_var].values

    if y.sum() < 5 or (len(y) - y.sum()) < 5:
        return {"tau": np.nan, "se": np.nan, "p_value": np.nan,
                "n": len(valid), "status": "no_outcome_variation"}

    try:
        result = rdrobust(y, x, c=cutoff)

        tau = result.coef.iloc[0]  # RD 估计值
        se = result.se.iloc[0]
        p_val = result.pv.iloc[0]
        ci_low = result.ci.iloc[0, 0]
        ci_high = result.ci.iloc[0, 1]
        bw = result.bws.iloc[0, 0]  # 最优带宽
        n_left = int(np.sum(x < cutoff))
        n_right = int(np.sum(x >= cutoff))

        return {
            "tau": tau,
            "se": se,
            "p_value": p_val,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "bandwidth": bw,
            "n_left": n_left,
            "n_right": n_right,
            "n": len(valid),
            "status": "ok",
        }
    except Exception as e:
        return {"tau": np.nan, "se": np.nan, "p_value": np.nan,
                "n": len(valid), "status": f"error: {str(e)[:80]}"}


def density_test(x: np.ndarray, cutoff: float) -> dict:
    """
    McCrary 密度检验: 检查运行变量在断点处是否有操纵
    如果断点处密度不连续，说明有选择性操纵，RDD 假设违反
    """
    # 简化密度检验: 比较断点两侧样本密度比例
    bw = np.std(x) * 0.5
    n_left = int(np.sum((x >= cutoff - bw) & (x < cutoff)))
    n_right = int(np.sum((x >= cutoff) & (x < cutoff + bw)))
    n_total = n_left + n_right
    if n_total > 10:
        ratio = n_left / n_total
        from scipy.stats import norm
        z = (ratio - 0.5) / np.sqrt(0.25 / n_total)
        p_val = 2 * (1 - norm.cdf(abs(z)))
        return {"density_p": p_val, "density_ratio": ratio, "status": "approx"}
    return {"density_p": np.nan, "status": "insufficient"}


def run(raw_all: pd.DataFrame):
    """对每个治疗变量在临床阈值处运行 RDD"""
    out_dir = OUTPUT_ROOT / "counterfactual"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = raw_all.copy()
    df["is_preeclampsia"] = (df["outcome"] == "preeclampsia").astype(int)
    df["is_gdm"] = (df["outcome"] == "gdm").astype(int)
    df["is_hypertension"] = (df["outcome"] == "hypertension").astype(int)
    df = df[df["gestational_week"] > 0].copy()

    # 子采样加速 rdrobust（保留断点附近数据密度）
    if len(df) > 50000:
        df = df.sample(n=50000, random_state=42)

    results = []
    for treat_name, (feat, op, cutoff) in TREATMENT_THRESHOLDS.items():
        if feat not in df.columns:
            continue

        for outcome_col in ["is_preeclampsia", "is_hypertension", "is_gdm"]:
            # 全样本 RDD
            print(f"  RDD: {feat} (cutoff={cutoff}) → {outcome_col}")
            res = rdd_estimate(df, feat, outcome_col, cutoff)
            res.update({"treatment": treat_name, "feature": feat,
                        "cutoff": cutoff, "outcome": outcome_col, "scope": "all"})

            # 密度检验
            x_valid = df[feat].dropna().values
            density = density_test(x_valid, cutoff)
            res["density_p"] = density.get("density_p", np.nan)

            results.append(res)

            # 按孕周窗口
            for gw_lo, gw_hi in GW_WINDOWS:
                wdf = df[(df["gestational_week"] >= gw_lo) & (df["gestational_week"] < gw_hi)]
                if len(wdf) < 200:
                    continue
                res_w = rdd_estimate(wdf, feat, outcome_col, cutoff)
                res_w.update({"treatment": treat_name, "feature": feat,
                              "cutoff": cutoff, "outcome": outcome_col,
                              "scope": f"{gw_lo}-{gw_hi}w"})
                res_w["density_p"] = np.nan  # 窗口内不做密度检验
                results.append(res_w)

    result_df = pd.DataFrame(results)
    result_df.to_csv(out_dir / "rdd_results.csv", index=False)

    # 确保数值列类型正确
    for col in ["tau", "se", "p_value", "ci_low", "ci_high", "bandwidth", "density_p"]:
        if col in result_df.columns:
            result_df[col] = pd.to_numeric(result_df[col], errors="coerce")

    # 汇总
    ok = result_df[result_df["status"] == "ok"]
    print(f"\n  RDD 分析完成: {len(ok)}/{len(result_df)} 成功")
    if len(ok) > 0:
        sig = ok[ok["p_value"] < 0.05]
        print(f"  显著 (p<0.05): {len(sig)}")

        pe_sig = sig[sig["outcome"] == "is_preeclampsia"]
        if len(pe_sig) > 0:
            print("\n  PE 显著 RDD 结果:")
            for _, r in pe_sig.sort_values("p_value").head(10).iterrows():
                print(f"    {r['feature']} (c={r['cutoff']}) @ {r['scope']}: "
                      f"τ={float(r['tau']):.4f} (p={float(r['p_value']):.4f}), "
                      f"bw={float(r.get('bandwidth', np.nan)):.2f}")

    return result_df

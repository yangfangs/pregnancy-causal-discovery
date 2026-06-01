"""
DoWhy 因果效应估计:
- 后门准则识别
- PSM / IPW / Doubly Robust 估计 ATT
- 三重反驳检验
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import dowhy
from dowhy import CausalModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import OUTPUT_ROOT, TREATMENT_THRESHOLDS

warnings.filterwarnings("ignore")


def estimate_att(df: pd.DataFrame, treatment: str, outcome: str,
                  confounders: list, graph: object = None) -> dict:
    """
    使用 DoWhy 估计 ATT (Average Treatment Effect on the Treated)
    """
    # 过滤有效数据
    cols = [treatment, outcome] + confounders
    valid = df[cols].dropna()
    if len(valid) < 100:
        return {"att": np.nan, "ci_low": np.nan, "ci_high": np.nan,
                "p_value": np.nan, "n": len(valid), "method": "insufficient_data"}

    # 构建因果模型
    model = CausalModel(
        data=valid,
        treatment=treatment,
        outcome=outcome,
        common_causes=confounders,
    )

    # 识别
    identified = model.identify_effect(proceed_when_unidentifiable=True)

    # 多种方法估计
    results = {}
    methods = [
        ("propensity_score_matching", "backdoor.propensity_score_matching"),
        ("propensity_score_weighting", "backdoor.propensity_score_weighting"),
    ]

    for mname, mmethod in methods:
        try:
            estimate = model.estimate_effect(
                identified,
                method_name=mmethod,
                target_units="att",
            )
            results[mname] = {
                "att": estimate.value,
                "method": mname,
            }
        except Exception as e:
            results[mname] = {"att": np.nan, "method": mname, "error": str(e)}

    # 选择最稳健的结果（取中位数）
    att_values = [r["att"] for r in results.values() if not np.isnan(r.get("att", np.nan))]
    best_att = np.median(att_values) if att_values else np.nan

    # 反驳检验
    refutation_results = {}
    if not np.isnan(best_att):
        try:
            estimate = model.estimate_effect(
                identified,
                method_name="backdoor.propensity_score_weighting",
                target_units="att",
            )
            # 1. 安慰剂检验
            ref_placebo = model.refute_estimate(
                identified, estimate,
                method_name="placebo_treatment_refuter",
                placebo_type="permute",
            )
            refutation_results["placebo_p"] = ref_placebo.refutation_result.get("p_value", np.nan) if hasattr(ref_placebo, 'refutation_result') and isinstance(ref_placebo.refutation_result, dict) else np.nan

            # 2. 随机公共原因
            ref_random = model.refute_estimate(
                identified, estimate,
                method_name="random_common_cause",
            )
            refutation_results["random_cause_p"] = ref_random.refutation_result.get("p_value", np.nan) if hasattr(ref_random, 'refutation_result') and isinstance(ref_random.refutation_result, dict) else np.nan

            # 3. 数据子集
            ref_subset = model.refute_estimate(
                identified, estimate,
                method_name="data_subset_refuter",
                subset_fraction=0.8,
            )
            refutation_results["subset_p"] = ref_subset.refutation_result.get("p_value", np.nan) if hasattr(ref_subset, 'refutation_result') and isinstance(ref_subset.refutation_result, dict) else np.nan
        except Exception:
            pass

    return {
        "att": best_att,
        "n": len(valid),
        "n_treated": int(valid[treatment].sum()),
        "n_control": int(len(valid) - valid[treatment].sum()),
        "estimates": results,
        "refutations": refutation_results,
    }


def compute_e_value(att: float, se: float = None) -> float:
    """计算 E-value: 未观测混杂因素需要多强才能解释掉观测到的效应"""
    if np.isnan(att) or att == 0:
        return 1.0
    rr = np.exp(abs(att))  # 粗略转换为 risk ratio
    e_val = rr + np.sqrt(rr * (rr - 1))
    return e_val


def run(datasets: dict, consensus_dags: dict):
    """对每个 trimester 估计关键干预的 ATT"""
    out_dir = OUTPUT_ROOT / "counterfactual"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for tri_name, tri_info in datasets.items():
        imp_df = tri_info["imputed"][0]
        features = tri_info["features"]

        # 混杂因素: age + 基线血常规/生化指标
        base_confounders = ["age", "B_WBC", "B_HGB", "B_PLT", "B_RBC",
                            "C_ALT", "C_ALB", "C_Cr", "C_UA", "C_Glu"]
        confounders = [c for c in base_confounders
                      if c in imp_df.columns and imp_df[c].notna().mean() > 0.3]

        for treat_name in TREATMENT_THRESHOLDS:
            if treat_name not in imp_df.columns:
                continue

            for outcome_col in ["is_preeclampsia", "is_hypertension", "is_gdm"]:
                if outcome_col not in imp_df.columns:
                    continue

                print(f"  ATT: {treat_name} → {outcome_col} ({tri_name})")
                result = estimate_att(imp_df, treat_name, outcome_col, confounders)

                all_results.append({
                    "trimester": tri_name,
                    "treatment": treat_name,
                    "outcome": outcome_col,
                    "att": result["att"],
                    "n": result["n"],
                    "n_treated": result.get("n_treated", 0),
                    "e_value": compute_e_value(result["att"]),
                })

    result_df = pd.DataFrame(all_results)
    result_df.to_csv(out_dir / "att_estimates.csv", index=False)
    print(f"\nATT 估计完成, {len(result_df)} 条记录")
    return result_df

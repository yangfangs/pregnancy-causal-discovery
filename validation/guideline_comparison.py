"""
临床指南对照验证:
将因果发现结果与 ACOG/NICE 指南对比
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import OUTPUT_ROOT

# ACOG/NICE 临床指南关键建议
GUIDELINES = [
    {
        "source": "ACOG Practice Bulletin #222 (2020)",
        "recommendation": "高危孕妇 12-28 周开始低剂量阿司匹林预防 PE",
        "timing": "12-28w",
        "gw_range": (12, 28),
        "relevant_markers": ["D_Dimer", "B_PLT", "FIB"],
        "mechanism": "抗凝/抗血小板",
    },
    {
        "source": "NICE NG133 (2019)",
        "recommendation": "20 周后监测血压和蛋白尿",
        "timing": "≥20w",
        "gw_range": (20, 40),
        "relevant_markers": ["U_PRO", "C_Cr", "C_UA"],
        "mechanism": "肾功监测",
    },
    {
        "source": "ACOG Practice Bulletin #222",
        "recommendation": "肝酶 + 血小板监测用于 HELLP 早期识别",
        "timing": "≥20w",
        "gw_range": (20, 40),
        "relevant_markers": ["C_ALT", "C_AST", "B_PLT", "C_LDH"],
        "mechanism": "肝功-血小板级联",
    },
    {
        "source": "WHO (2021)",
        "recommendation": "妊娠期补钙预防 PE (低钙饮食人群)",
        "timing": "全孕期",
        "gw_range": (0, 40),
        "relevant_markers": ["C_Ca"],
        "mechanism": "电解质调节",
    },
    {
        "source": "ISSHP (2018)",
        "recommendation": "尿酸升高是 PE 严重程度标志",
        "timing": "≥20w",
        "gw_range": (20, 40),
        "relevant_markers": ["C_UA"],
        "mechanism": "肾功标志物",
    },
]


def compare_with_guidelines(window_results: pd.DataFrame,
                              cascade_results: dict,
                              edge_comparison: pd.DataFrame = None) -> pd.DataFrame:
    """
    将我们的因果发现与临床指南对比
    """
    out_dir = OUTPUT_ROOT / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for gl in GUIDELINES:
        concordance_notes = []

        # 检查窗口 ATT
        if window_results is not None and len(window_results) > 0:
            for marker in gl["relevant_markers"]:
                treat_name = None
                for tn in window_results["treatment"].unique():
                    if marker.lower() in tn.lower():
                        treat_name = tn
                        break
                if treat_name is None:
                    continue

                relevant = window_results[
                    (window_results["treatment"] == treat_name)
                    & (window_results["gw_start"] >= gl["gw_range"][0])
                    & (window_results["gw_end"] <= gl["gw_range"][1])
                    & (window_results["outcome"] == "is_preeclampsia")
                ]
                if len(relevant) > 0:
                    max_att = relevant["att"].abs().max()
                    concordance_notes.append(f"{treat_name}: max|ATT|={max_att:.4f}")

        # 检查级联中是否包含相关标志物
        if cascade_results:
            for tri, df in cascade_results.items():
                if df is None or len(df) == 0:
                    continue
                for marker in gl["relevant_markers"]:
                    matches = df[df["path"].str.contains(marker, na=False)]
                    if len(matches) > 0:
                        concordance_notes.append(f"{marker} in {tri} cascades ({len(matches)})")

        records.append({
            "guideline_source": gl["source"],
            "recommendation": gl["recommendation"],
            "timing": gl["timing"],
            "our_findings": "; ".join(concordance_notes) if concordance_notes else "无直接对应",
            "concordance": "是" if concordance_notes else "待验证",
        })

    result = pd.DataFrame(records)
    result.to_csv(out_dir / "guideline_comparison.csv", index=False)

    print("\n=== 指南对照 ===")
    for _, row in result.iterrows():
        status = "✓" if row["concordance"] == "是" else "?"
        print(f"  {status} {row['guideline_source']}: {row['recommendation']}")
        print(f"    发现: {row['our_findings']}")

    return result


def run(window_results, cascade_results, edge_comparison=None):
    return compare_with_guidelines(window_results, cascade_results, edge_comparison)

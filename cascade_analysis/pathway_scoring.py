"""
路径评分:
- 统计强度 (边置信度乘积)
- 临床合理性评分
- 跨中心复现率
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import OUTPUT_ROOT

# 已知生物学机制的因果边（文献支持）
KNOWN_MECHANISMS = {
    ("D_Dimer", "FIB"): "凝血激活与纤维蛋白原消耗",
    ("D_Dimer", "C_ALT"): "微血栓导致肝缺血损伤",
    ("B_PLT", "D_Dimer"): "血小板消耗伴凝血激活",
    ("C_ALT", "C_AST"): "肝细胞损伤",
    ("C_ALT", "C_TBIL"): "肝功异常→胆红素升高",
    ("C_UA", "U_PRO"): "肾功损伤→蛋白尿",
    ("C_Cr", "U_PRO"): "肾小球滤过受损",
    ("C_Glu", "C_TG"): "糖脂代谢关联",
    ("B_WBC", "B_NEU_pct"): "炎症反应",
    ("C_ALB", "U_PRO"): "低蛋白血症与蛋白尿",
    ("FIB", "D_Dimer"): "纤维蛋白溶解产物",
    ("C_UA", "C_Cr"): "尿酸与肌酐协同升高",
    ("B_PLT", "FIB"): "凝血-血小板交互",
}


def score_pathway(path_nodes: list, cascade_df: pd.DataFrame) -> dict:
    """为一条路径计算多维评分"""
    edges = [(path_nodes[i], path_nodes[i + 1]) for i in range(len(path_nodes) - 1)]

    # 临床合理性评分: 每个有文献支持的边 +1
    plausibility = sum(1 for e in edges if e in KNOWN_MECHANISMS)

    return {
        "n_known_edges": plausibility,
        "plausibility_ratio": plausibility / len(edges) if edges else 0,
    }


def run(all_cascades: dict):
    """对所有 trimester 的级联路径进行评分"""
    out_dir = OUTPUT_ROOT / "cascades"

    scored_all = {}
    for tri_name, cascade_df in all_cascades.items():
        if cascade_df is None or len(cascade_df) == 0:
            continue

        scores = []
        for _, row in cascade_df.iterrows():
            nodes = row["nodes"] if isinstance(row["nodes"], list) else row["path"].split(" → ")
            sc = score_pathway(nodes, cascade_df)
            scores.append(sc)

        score_df = pd.DataFrame(scores)
        result = pd.concat([cascade_df.reset_index(drop=True), score_df], axis=1)

        # 综合排序: strength * (1 + plausibility_ratio)
        result["composite_score"] = result["strength"] * (1 + result["plausibility_ratio"])
        result = result.sort_values("composite_score", ascending=False)

        result.to_csv(out_dir / f"{tri_name}_scored_cascades.csv", index=False)
        scored_all[tri_name] = result

        print(f"\n{tri_name} Top scored cascades:")
        for _, row in result.head(5).iterrows():
            print(f"  {row['path']} | score={row['composite_score']:.3f} | known={row['n_known_edges']}")

    return scored_all

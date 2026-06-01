"""临床特征分组 — 用于因果图解读和搜索空间约束"""

CLINICAL_GROUPS = {
    "coagulation": {
        "label": "凝血系统",
        "color": "#E64B35",
        "features": ["PT", "APTT", "TT", "FIB", "INR", "D_Dimer", "FDP"],
    },
    "liver": {
        "label": "肝功能",
        "color": "#4DBBD5",
        "features": ["C_ALT", "C_AST", "C_GGT", "C_ALP", "C_TBIL", "C_DBIL", "C_LDH",
                      "C_TP", "C_ALB", "C_GLB"],
    },
    "renal": {
        "label": "肾功能",
        "color": "#00A087",
        "features": ["C_BUN", "C_Cr", "C_UA"],
    },
    "inflammatory": {
        "label": "炎症标志物",
        "color": "#3C5488",
        "features": ["B_WBC", "B_NEU_pct", "B_NEU_abs", "B_LYM_pct", "B_LYM_abs",
                      "B_MONO_pct", "B_EOS_pct", "B_BASO_pct"],
    },
    "metabolic": {
        "label": "代谢指标",
        "color": "#8491B4",
        "features": ["C_Glu", "C_TG", "C_TC", "C_HDL", "C_LDL"],
    },
    "hematologic": {
        "label": "血液系统",
        "color": "#F39B7F",
        "features": ["B_RBC", "B_HGB", "B_HCT", "B_PLT", "B_MCV", "B_MCH", "B_MCHC", "B_RDW"],
    },
    "electrolytes": {
        "label": "电解质",
        "color": "#B09C85",
        "features": ["C_K", "C_Na", "C_Cl", "C_Ca", "C_P", "C_Mg"],
    },
    "urine": {
        "label": "尿液指标",
        "color": "#91D1C2",
        "features": ["U_SG", "U_pH", "U_LEU", "U_ERY", "U_PRO", "U_GLU", "U_KET",
                      "U_UBG", "U_BIL", "U_Bacteria"],
    },
}


def feature_to_group(feat: str) -> str:
    """返回特征所属的临床分组名称"""
    for gname, ginfo in CLINICAL_GROUPS.items():
        if feat in ginfo["features"]:
            return gname
    return "other"


def feature_to_color(feat: str) -> str:
    """返回特征对应的颜色"""
    grp = feature_to_group(feat)
    return CLINICAL_GROUPS.get(grp, {}).get("color", "#999999")

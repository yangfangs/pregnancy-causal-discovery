"""中心配置、路径、特征定义、阈值参数"""

import os
import sys
from pathlib import Path

# ── 路径 ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Set CAUSAL_DATA_ROOT env var to your cleaned-data directory, or pass --data-root
# to the pipeline. Defaults to data/ inside the project for the example dataset.
DATA_ROOT = Path(os.environ.get("CAUSAL_DATA_ROOT", PROJECT_ROOT / "data"))
OUTPUT_ROOT = PROJECT_ROOT / "results"

PYTHON = sys.executable

# 每个中心的 parquet 路径
DATA_PATHS = {
    "SFY": {
        "blood": DATA_ROOT / "SFY" / "BloodRoutine_cleaned.parquet",
        "biochem": DATA_ROOT / "SFY" / "Biochemistry_cleaned.parquet",
        "coag": DATA_ROOT / "SFY" / "Coagulation_cleaned.parquet",
        "urine": DATA_ROOT / "SFY" / "UrineRoutine_cleaned.parquet",
    },
    "AY": {
        "blood": DATA_ROOT / "AY" / "BloodRoutine.parquet",
        "biochem": DATA_ROOT / "AY" / "Biochemistry.parquet",
        "coag": DATA_ROOT / "AY" / "Coagulation.parquet",
        "urine": DATA_ROOT / "AY" / "UrineRoutine.parquet",
    },
    "PQ": {
        "blood": DATA_ROOT / "PQ" / "BloodRoutine.parquet",
        "biochem": DATA_ROOT / "PQ" / "Biochemistry.parquet",
        "coag": DATA_ROOT / "PQ" / "Coagulation.parquet",
        "urine": DATA_ROOT / "PQ" / "UrineRoutine.parquet",
    },
    "XX": {
        "blood": DATA_ROOT / "XX" / "BloodRoutine_cleaned.parquet",
        "biochem": DATA_ROOT / "XX" / "Biochemistry_cleaned.parquet",
        "coag": DATA_ROOT / "XX" / "Coagulation_cleaned.parquet",
        "urine": DATA_ROOT / "XX" / "UrineRoutine_cleaned.parquet",
    },
}

CENTERS = ["SFY", "AY", "PQ", "XX"]

# ── 特征映射（每中心列名 → 统一字段）──
FEATURE_MAPPING = {
    "patient_id": {"SFY": "病历号", "AY": "IDCard", "PQ": "IDCard", "XX": "就诊卡号"},
    "gestational_week": {
        "SFY": "GestationalWeeks",
        "AY": "GestationalWeek",
        "PQ": "GestationalWeek",
        "XX": "GestationalWeeks",
    },
    "age": {"SFY": "年龄", "AY": "Age", "PQ": ["Age_x", "Age_y", "Age"], "XX": "年龄"},
    "outcome": {
        "SFY": "PregnancyOutcome",
        "AY": "PregnancyOutcome",
        "PQ": "PregnancyOutcome",
        "XX": "PregnancyOutcome",
    },
}

OUTCOME_MAP = {
    "妊娠期糖尿病": "gdm",
    "糖尿病合并妊娠": "gdm",
    "GDM": "gdm",
    "妊娠期高血压": "hypertension",
    "慢性高血压并发妊娠": "hypertension",
    "高血压合并妊娠": "hypertension",
    "重度子痫前期": "preeclampsia",
    "轻度子痫前期": "preeclampsia",
    "子痫前期": "preeclampsia",
    "重度先兆子痫": "preeclampsia",
    "正常妊娠": "normal",
    "正常分娩": "normal",
    "Normal": "normal",
}

# 结局优先级（同患者同周取最严重）
OUTCOME_PRIORITY = {"preeclampsia": 4, "hypertension": 3, "gdm": 2, "normal": 1, "unknown": 0}

# ── 标准化特征 ──
COAG_FEATURES = ["PT", "APTT", "TT", "FIB", "INR", "D_Dimer", "FDP"]
BLOOD_FEATURES = [
    "B_WBC", "B_RBC", "B_HGB", "B_HCT", "B_PLT", "B_MCV", "B_MCH", "B_MCHC",
    "B_NEU_pct", "B_LYM_pct", "B_MONO_pct", "B_EOS_pct", "B_BASO_pct",
    "B_NEU_abs", "B_LYM_abs", "B_RDW",
]
BIOCHEM_FEATURES = [
    "C_ALT", "C_AST", "C_GGT", "C_ALP", "C_TP", "C_ALB", "C_GLB",
    "C_TBIL", "C_DBIL", "C_BUN", "C_Cr", "C_UA", "C_Glu",
    "C_TG", "C_TC", "C_HDL", "C_LDL", "C_K", "C_Na", "C_Cl",
    "C_Ca", "C_P", "C_Mg", "C_LDH",
]
URINE_FEATURES = [
    "U_SG", "U_pH", "U_LEU", "U_ERY", "U_PRO", "U_GLU", "U_KET",
    "U_UBG", "U_BIL", "U_Bacteria",
]
ALL_FEATURES = COAG_FEATURES + BLOOD_FEATURES + BIOCHEM_FEATURES + URINE_FEATURES

# ── 孕期分层 ──
TRIMESTER_BOUNDS = {"T1": (0, 14), "T2": (14, 28), "T3": (28, 46)}

# ── 因果发现参数 ──
MIN_FEATURE_COVERAGE = 0.25  # 特征纳入最低非空率（降低以纳入凝血特征）
MIN_STRATUM_SIZE = 200       # 每层最小样本量
PC_ALPHA_SWEEP = [0.001, 0.005, 0.01, 0.05]
BOOTSTRAP_N = 500            # bootstrap 次数
ENSEMBLE_THRESHOLD = 0.50    # 集成边纳入阈值

# ── 反事实参数 ──
# 二值化治疗变量的临床阈值（原始值，非标准化）
TREATMENT_THRESHOLDS = {
    "D_Dimer_high": ("D_Dimer", ">", 0.5),
    "PLT_low": ("B_PLT", "<", 150),
    "ALT_high": ("C_ALT", ">", 40),
    "UA_high": ("C_UA", ">", 360),
    "PRO_positive": ("U_PRO", ">=", 1),
    "FIB_high": ("FIB", ">", 4.0),
}
GW_WINDOWS = [(8, 12), (12, 16), (16, 20), (20, 24), (24, 28), (28, 32), (32, 36), (36, 40)]

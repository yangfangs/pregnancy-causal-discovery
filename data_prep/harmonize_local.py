"""
多模态数据标准化 (Temporal) — 本地路径版
改自 Cleaned_Data_all/run_harmonization_temporal.py，修正路径并输出到 results/harmonized/
同时输出原始值（raw）和标准化值（scaled）两套 parquet。
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import (
    ALL_FEATURES, BIOCHEM_FEATURES, BLOOD_FEATURES, COAG_FEATURES,
    DATA_PATHS, CENTERS, FEATURE_MAPPING, OUTCOME_MAP, OUTCOME_PRIORITY,
    OUTPUT_ROOT, URINE_FEATURES,
)

warnings.filterwarnings("ignore")


def clean_numeric(series: pd.Series) -> pd.Series:
    if series.dtype == "object" or series.dtype.name == "category":
        series = series.astype(str)
        qualitative_map = {
            "-": "0", "－": "0", "阴性": "0", "negative": "0",
            "+-": "0.5", "±": "0.5", "弱阳性": "0.5",
            "+": "1", "1+": "1", "阳性": "1", "positive": "1",
            "++": "2", "2+": "2",
            "+++": "3", "3+": "3",
            "++++": "4", "4+": "4",
        }
        mapped = series.map(qualitative_map)
        series = series.where(mapped.isna(), mapped)
        series = series.str.replace(r"[<>=≥≤]", "", regex=True)
        series = series.str.replace(r"[^\d.\-]", "", regex=True)
    return pd.to_numeric(series, errors="coerce")


def get_patient_id(df: pd.DataFrame, center: str) -> pd.Series:
    if center == "XX":
        def _gen(row):
            for col in ["身份证/护照号", "身份证号"]:
                if col in row.index:
                    v = row[col]
                    if pd.notna(v) and str(v).strip() not in ("", "nan", "None", "None.0"):
                        return f"ID_{str(v).strip()}"
            if "就诊卡号" in row.index:
                v = row["就诊卡号"]
                if pd.notna(v) and str(v).strip() not in ("", "nan", "None", "None.0"):
                    return f"CARD_{str(v).replace('.0', '').strip()}"
            name = str(row.get("姓名", "")).strip() if pd.notna(row.get("姓名")) else ""
            age = str(row.get("年龄", "")).strip() if pd.notna(row.get("年龄")) else ""
            if name and name not in ("", "nan", "None"):
                return f"NAME_{name}_{age}"
            return None
        ids = df.apply(_gen, axis=1)
        null_mask = ids.isna()
        if null_mask.sum() > 0:
            ids.loc[null_mask] = [f"UNK_{i}" for i in range(null_mask.sum())]
        return ids

    col = FEATURE_MAPPING["patient_id"][center]
    if col in df.columns:
        return df[col].astype(str)
    return pd.Series(["unknown"] * len(df), index=df.index)


def get_age(df: pd.DataFrame, center: str) -> pd.Series:
    cols = FEATURE_MAPPING["age"][center]
    if isinstance(cols, list):
        for c in cols:
            if c in df.columns and df[c].notna().any():
                return clean_numeric(df[c])
    elif cols in df.columns:
        return clean_numeric(df[cols])
    return pd.Series([np.nan] * len(df), index=df.index)


def get_gestational_week(df: pd.DataFrame, center: str) -> pd.Series:
    col = FEATURE_MAPPING["gestational_week"][center]
    if col in df.columns:
        return clean_numeric(df[col])
    return pd.Series([np.nan] * len(df), index=df.index)


def get_outcome(df: pd.DataFrame, center: str) -> pd.Series:
    col = FEATURE_MAPPING["outcome"][center]
    if col not in df.columns:
        return pd.Series(["unknown"] * len(df), index=df.index)
    return df[col].fillna("unknown").astype(str).map(lambda x: OUTCOME_MAP.get(x, "unknown"))


def load_modality(center: str, modality: str):
    path = DATA_PATHS[center].get(modality)
    if path is None or not Path(path).exists():
        return None
    try:
        return pd.read_parquet(path)
    except Exception:
        return None


def extract_features(df, center, feat_map, modality_name):
    result = pd.DataFrame(index=df.index)
    result["patient_id"] = get_patient_id(df, center)
    result["age"] = get_age(df, center)
    result["gestational_week"] = get_gestational_week(df, center)
    result["outcome"] = get_outcome(df, center)
    for feat, candidates in feat_map.items():
        for col in candidates:
            if col in df.columns:
                result[feat] = clean_numeric(df[col])
                break
        if feat not in result.columns:
            result[feat] = np.nan
    return result


# ── 每个模态的特征候选映射 ──
COAG_MAP = {
    "PT": ["PT", "PT（凝血酶原时间）"],
    "APTT": ["APTT", "APTT（活化部分凝血活酶时间）"],
    "TT": ["TT", "TT（凝血酶时间）"],
    "FIB": ["FIB", "FIB（纤维蛋白原）", "Fbg"],
    "INR": ["INR"],
    "D_Dimer": ["D-Dimer", "D-二聚体", "DD（D-二聚体）", "D-D"],
    "FDP": ["FDP", "FDP（纤维蛋白降解产物）"],
}
BLOOD_MAP = {
    "B_WBC": ["WBC"], "B_RBC": ["RBC"], "B_HGB": ["HGB", "Hb"], "B_HCT": ["HCT"],
    "B_PLT": ["PLT"], "B_MCV": ["MCV"], "B_MCH": ["MCH"], "B_MCHC": ["MCHC"],
    "B_NEU_pct": ["NEUT%", "NEU%", "Neu%"], "B_LYM_pct": ["LYMPH%", "LYM%", "Lym%"],
    "B_MONO_pct": ["MONO%", "Mon%"], "B_EOS_pct": ["EO%", "Eos%"],
    "B_BASO_pct": ["BASO%", "Bas%"], "B_NEU_abs": ["NEUT#", "NEU#"],
    "B_LYM_abs": ["LYMPH#", "LYM#"], "B_RDW": ["RDW-CV", "RDW"],
}
BIOCHEM_MAP = {
    "C_ALT": ["ALT"], "C_AST": ["AST"], "C_GGT": ["GGT"], "C_ALP": ["ALP"],
    "C_TP": ["TP"], "C_ALB": ["ALB"], "C_GLB": ["GLB"],
    "C_TBIL": ["TBIL"], "C_DBIL": ["DBIL"],
    "C_BUN": ["UREA", "Urea", "BUN"], "C_Cr": ["CREA", "Cr"], "C_UA": ["UA"],
    "C_Glu": ["GLU", "Glu"], "C_TG": ["TG"], "C_TC": ["TC", "CHOL"],
    "C_HDL": ["HDL", "HDL-C"], "C_LDL": ["LDL", "LDL-C"],
    "C_K": ["K"], "C_Na": ["Na", "NA"], "C_Cl": ["Cl", "CL"],
    "C_Ca": ["Ca", "CA"], "C_P": ["P"], "C_Mg": ["Mg"], "C_LDH": ["LDH"],
}
URINE_MAP = {
    "U_SG": ["U_SG", "SG"], "U_pH": ["U_PH", "pH", "U_pH"],
    "U_LEU": ["U_WBC_Others", "LEU", "U_LEU", "U_WBC_Micro"],
    "U_ERY": ["U_ERY", "ERY"], "U_PRO": ["U_PRO", "PRO"],
    "U_GLU": ["U_GLU"], "U_KET": ["U_KET", "KET"],
    "U_UBG": ["U_UBG", "UBG"], "U_BIL": ["U_BIL", "BIL"],
    "U_Bacteria": ["U_Bacteria", "Bacteria", "细菌"],
}


def process_center(center: str) -> pd.DataFrame:
    print(f"\n处理 {center} ...")
    modality_data = []

    for mod_name, mod_map in [("coag", COAG_MAP), ("blood", BLOOD_MAP),
                               ("biochem", BIOCHEM_MAP), ("urine", URINE_MAP)]:
        df = load_modality(center, mod_name)
        if df is not None:
            modality_data.append(extract_features(df, center, mod_map, mod_name))

    if not modality_data:
        return pd.DataFrame()

    combined = pd.concat(modality_data, ignore_index=True)
    combined["gestational_week"] = combined["gestational_week"].fillna(-1)
    combined["gestational_week"] = combined["gestational_week"].round().astype(int)
    combined = combined[
        (combined["gestational_week"] == -1)
        | ((combined["gestational_week"] >= 0) & (combined["gestational_week"] <= 45))
    ]

    # 按 (patient_id, gestational_week) 聚合
    numeric_cols = [c for c in ALL_FEATURES + ["age"] if c in combined.columns]
    agg_dict = {col: "mean" for col in numeric_cols}
    combined["outcome_score"] = combined["outcome"].map(OUTCOME_PRIORITY).fillna(0)
    agg_dict["outcome_score"] = "max"

    result = combined.groupby(["patient_id", "gestational_week"]).agg(agg_dict).reset_index()
    inv = {v: k for k, v in OUTCOME_PRIORITY.items()}
    result["outcome"] = result["outcome_score"].map(inv)
    result.drop(columns=["outcome_score"], inplace=True)
    result["center"] = center

    for feat in ALL_FEATURES:
        if feat not in result.columns:
            result[feat] = np.nan

    print(f"  {center}: {len(result)} records")
    return result


def run():
    out_dir = OUTPUT_ROOT / "harmonized"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_data = {}
    for center in CENTERS:
        df = process_center(center)
        if len(df) > 0:
            all_data[center] = df

    # ── 保存原始值（raw）──
    for center, df in all_data.items():
        meta_cols = ["patient_id", "center", "age", "gestational_week", "outcome"]
        raw_df = df[meta_cols + ALL_FEATURES].copy()
        raw_df.to_parquet(out_dir / f"raw_{center}.parquet", index=False)
        print(f"  RAW {center}: {len(raw_df)} -> {out_dir / f'raw_{center}.parquet'}")

    # ── 标准化（以 SFY 为基准）──
    sfy_df = all_data["SFY"]
    scaler = StandardScaler()
    sfy_filled = sfy_df[ALL_FEATURES].fillna(sfy_df[ALL_FEATURES].median())
    scaler.fit(sfy_filled)

    for center, df in all_data.items():
        feats = df[ALL_FEATURES].copy()
        filled = feats.fillna(sfy_filled.median())
        scaled = pd.DataFrame(scaler.transform(filled), columns=ALL_FEATURES, index=df.index)
        for col in ALL_FEATURES:
            scaled.loc[feats[col].isna(), col] = np.nan

        meta = df[["patient_id", "center", "age", "gestational_week", "outcome"]].copy()
        final = pd.concat([meta.reset_index(drop=True), scaled.reset_index(drop=True)], axis=1)
        final.to_parquet(out_dir / f"harmonized_{center}.parquet", index=False)
        print(f"  SCALED {center}: {len(final)} -> {out_dir / f'harmonized_{center}.parquet'}")

    # 保存 scaler 参数
    scaler_df = pd.DataFrame({"feature": ALL_FEATURES, "mean": scaler.mean_, "std": scaler.scale_})
    scaler_df.to_csv(out_dir / "scaler_params.csv", index=False)
    print("\n数据标准化完成!")
    return all_data


if __name__ == "__main__":
    run()

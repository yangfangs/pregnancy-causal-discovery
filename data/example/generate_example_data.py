"""
Generate synthetic example dataset for code testing.

Produces:
  * data/example/example_cohort.parquet — a de-identified, fully synthetic
    dataset that mimics the column schema and approximate marginal
    distributions of the real cohort. No real patient information is included.
  * results/harmonized/{raw_SFY,harmonized_SFY}.parquet + scaler_params.csv —
    the harmonized-layout files the pipeline consumes. With these in place,
    `python pipeline/run_all.py` runs the whole framework on synthetic data
    (it detects the existing harmonized file and skips the data-prep phase that
    would otherwise require the restricted multi-center inputs).

Usage:
    python data/example/generate_example_data.py
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from config.settings import ALL_FEATURES  # noqa: E402

RNG = np.random.default_rng(42)
N = 500  # synthetic patients
OUT = Path(__file__).parent / "example_cohort.parquet"
HARMONIZED_DIR = PROJECT_ROOT / "results" / "harmonized"


def _rnorm(mu, sigma, size, lo=None, hi=None):
    x = RNG.normal(mu, sigma, size)
    if lo is not None:
        x = np.clip(x, lo, None)
    if hi is not None:
        x = np.clip(x, None, hi)
    return x


def main():
    outcome_labels = ["normal", "gdm", "hypertension", "preeclampsia"]
    outcome_probs = [0.82, 0.10, 0.05, 0.03]
    outcomes = RNG.choice(outcome_labels, size=N, p=outcome_probs)

    age = _rnorm(29, 4, N, lo=18, hi=48)
    gw = _rnorm(28, 10, N, lo=6, hi=42).astype(int)
    trimester = pd.cut(
        gw,
        bins=[0, 14, 28, 46],
        labels=["T1", "T2", "T3"],
        right=False,
    )

    # Coagulation (7)
    pt = _rnorm(12.5, 1.2, N, lo=8)
    aptt = _rnorm(30, 4, N, lo=20)
    tt = _rnorm(16, 2, N, lo=10)
    fib = _rnorm(3.5, 0.8, N, lo=1.5)
    inr = _rnorm(1.0, 0.1, N, lo=0.7)
    d_dimer = np.abs(_rnorm(0.4, 0.3, N))
    fdp = np.abs(_rnorm(3.0, 1.5, N))

    # elevate D-Dimer + FIB in PE cases
    pe_mask = outcomes == "preeclampsia"
    d_dimer[pe_mask] += RNG.uniform(0.3, 1.0, pe_mask.sum())
    fib[pe_mask] += RNG.uniform(0.5, 1.5, pe_mask.sum())

    # Blood routine (16) — simplified
    wbc = _rnorm(8.5, 2.0, N, lo=3)
    rbc = _rnorm(3.8, 0.4, N, lo=2.5)
    hgb = _rnorm(115, 12, N, lo=70)
    hct = _rnorm(0.35, 0.04, N, lo=0.2)
    plt = _rnorm(210, 50, N, lo=50)
    mcv = _rnorm(88, 7, N)
    mch = _rnorm(29, 3, N)
    mchc = _rnorm(330, 15, N)
    neu_pct = _rnorm(68, 8, N, lo=30, hi=95)
    lym_pct = 100 - neu_pct - _rnorm(8, 2, N, lo=2, hi=20)
    rdw = _rnorm(13.5, 1.0, N)

    # Biochemistry (10 of 24, representative)
    alt = np.abs(_rnorm(20, 10, N))
    ast = np.abs(_rnorm(22, 8, N))
    alp = _rnorm(95, 30, N, lo=30)
    alb = _rnorm(38, 3, N, lo=25)
    tbil = _rnorm(10, 4, N, lo=2)
    bun = _rnorm(4.5, 1.0, N, lo=1.5)
    cr = _rnorm(55, 12, N, lo=30)
    ua = _rnorm(290, 60, N, lo=120)
    glu = _rnorm(5.0, 1.0, N, lo=2.5)

    # elevate UA + ALT in PE
    ua[pe_mask] += RNG.uniform(50, 150, pe_mask.sum())
    alt[pe_mask] += RNG.uniform(10, 60, pe_mask.sum())

    # elevate Glu in GDM
    gdm_mask = outcomes == "gdm"
    glu[gdm_mask] += RNG.uniform(1.0, 3.5, gdm_mask.sum())

    # Urine routine (5 key)
    u_pro = RNG.choice([0, 0.5, 1, 2, 3], size=N, p=[0.80, 0.08, 0.07, 0.03, 0.02])
    u_pro[pe_mask] = RNG.choice([1, 2, 3], size=pe_mask.sum(), p=[0.4, 0.35, 0.25])
    u_ery = RNG.choice([0, 0.5, 1, 2], size=N, p=[0.75, 0.15, 0.07, 0.03])
    u_glu = RNG.choice([0, 1, 2], size=N, p=[0.85, 0.10, 0.05])
    u_leu = RNG.choice([0, 0.5, 1], size=N, p=[0.88, 0.08, 0.04])
    u_ph = _rnorm(5.5, 0.8, N, lo=4.5, hi=8.5)

    df = pd.DataFrame(
        {
            "patient_id": [f"SYN_{i:05d}" for i in range(N)],
            "gestational_week": gw,
            "trimester": trimester.astype(str),
            "age": age.round(1),
            "outcome": outcomes,
            # Coagulation
            "PT": pt.round(1),
            "APTT": aptt.round(1),
            "TT": tt.round(1),
            "FIB": fib.round(2),
            "INR": inr.round(2),
            "D_Dimer": d_dimer.round(3),
            "FDP": fdp.round(2),
            # Blood routine
            "B_WBC": wbc.round(2),
            "B_RBC": rbc.round(2),
            "B_HGB": hgb.round(1),
            "B_HCT": hct.round(3),
            "B_PLT": plt.round(0).astype(int),
            "B_MCV": mcv.round(1),
            "B_MCH": mch.round(1),
            "B_MCHC": mchc.round(0).astype(int),
            "B_NEU_pct": neu_pct.round(1),
            "B_LYM_pct": lym_pct.round(1),
            "B_RDW": rdw.round(1),
            # Biochemistry (subset)
            "C_ALT": alt.round(1),
            "C_AST": ast.round(1),
            "C_ALP": alp.round(1),
            "C_ALB": alb.round(1),
            "C_TBIL": tbil.round(1),
            "C_BUN": bun.round(2),
            "C_Cr": cr.round(1),
            "C_UA": ua.round(1),
            "C_Glu": glu.round(2),
            # Urine routine
            "U_PRO": u_pro,
            "U_ERY": u_ery,
            "U_GLU": u_glu,
            "U_LEU": u_leu,
            "U_pH": u_ph.round(1),
        }
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(f"Saved {len(df)} synthetic records to {OUT}")
    print(df.groupby("outcome").size())

    write_harmonized_layout(df)


def write_harmonized_layout(df: pd.DataFrame) -> None:
    """Emit the harmonized-layout files the pipeline consumes.

    Mirrors the column contract of data_prep/harmonize_local.run():
      raw_SFY.parquet        = meta + ALL_FEATURES (raw values)
      harmonized_SFY.parquet = meta + ALL_FEATURES (z-scored; NaN preserved)
      scaler_params.csv      = feature, mean, std

    Features the synthetic generator does not populate are written as all-NaN
    columns; the pipeline's coverage filter drops them automatically, exactly
    as it would for sparsely measured features in the real cohort.
    """
    # Safety guard: never silently clobber existing harmonized files (e.g. a real
    # cohort cached in results/harmonized/). Pass --force to overwrite.
    existing = HARMONIZED_DIR / "harmonized_SFY.parquet"
    if existing.exists() and "--force" not in sys.argv:
        print(f"WARNING: {existing} already exists — skipping harmonized-layout write.")
        print("  (Pass --force to overwrite. On a fresh clone this directory is empty "
              "and the files are written automatically.)")
        return

    meta_cols = ["patient_id", "center", "age", "gestational_week", "outcome"]
    full = df.copy()
    full["center"] = "SFY"  # single synthetic center; pipeline keys on SFY

    # Ensure every expected feature column exists (missing → NaN).
    for feat in ALL_FEATURES:
        if feat not in full.columns:
            full[feat] = np.nan

    raw = full[meta_cols + ALL_FEATURES].copy()

    # NaN-safe per-feature standardization (no sklearn dependency at gen time).
    means, stds, scaled = {}, {}, {}
    for feat in ALL_FEATURES:
        col = pd.to_numeric(raw[feat], errors="coerce")
        mu, sd = col.mean(), col.std(ddof=0)
        means[feat] = mu
        stds[feat] = sd
        # All-NaN or zero-variance columns stay NaN / centered without scaling.
        scaled[feat] = (col - mu) / sd if (sd is not None and sd > 0) else np.nan

    harmonized = raw[meta_cols].copy()
    for feat in ALL_FEATURES:
        harmonized[feat] = scaled[feat]

    HARMONIZED_DIR.mkdir(parents=True, exist_ok=True)
    raw.to_parquet(HARMONIZED_DIR / "raw_SFY.parquet", index=False)
    harmonized.to_parquet(HARMONIZED_DIR / "harmonized_SFY.parquet", index=False)
    pd.DataFrame(
        {"feature": ALL_FEATURES,
         "mean": [means[f] for f in ALL_FEATURES],
         "std": [stds[f] for f in ALL_FEATURES]}
    ).to_csv(HARMONIZED_DIR / "scaler_params.csv", index=False)
    print(f"Wrote harmonized-layout files to {HARMONIZED_DIR}/ "
          f"(raw_SFY, harmonized_SFY, scaler_params) — run `python pipeline/run_all.py`")


if __name__ == "__main__":
    main()

# A Framework for Temporal Causal Discovery in Pregnancy Complications

> Analysis framework accompanying:
> **"Causal Discovery and Identification of Interventable Windows for Pregnancy
> Complications: A Multi-Center Temporal Analysis"**

---

## Overview

This repository provides a reusable, end-to-end **framework** for trimester-stratified
causal discovery on routine clinical laboratory data, applied here to pregnancy
complications (pre-eclampsia, gestational hypertension, gestational diabetes mellitus).

The framework covers the full analysis path — from harmonizing multi-center laboratory
panels, through ensemble causal-graph learning, to counterfactual estimation of clinically
actionable intervention windows:

- **Multi-algorithm ensemble causal discovery** — PC, FCI, DirectLiNGAM, and NOTEARS-MLP,
  aggregated by bootstrap edge-frequency into a consensus DAG per trimester.
- **Cross-system causal cascade identification** — extraction and scoring of multi-step
  pathways across coagulation, hepatic, renal, and other systems.
- **Counterfactual intervention-window estimation** — PSW-ATT (DoWhy), IV-2SLS
  (linearmodels), and regression discontinuity (rdrobust) across gestational-week windows.
- **External replication** — cross-center reproducibility and guideline comparison utilities.

The repository ships the **method code** plus a **fully synthetic example dataset** so the
pipeline can be run end-to-end without access to patient data. Study-specific results,
figures, and the manuscript are not included (see [Scope](#scope)).

---

## Repository Structure

```
CausalDiscovery/
├── config/                  # Paths, feature definitions, thresholds
│   ├── settings.py          # CAUSAL_DATA_ROOT, feature groups, hyperparameters
│   └── feature_groups.py
├── data_prep/               # Data harmonization & dataset construction
│   ├── harmonize_local.py   # Column mapping + standardization across centers
│   ├── build_causal_dataset.py  # Trimester stratification, MICE imputation
│   └── feature_selection.py
├── causal_discovery/        # Causal graph learning
│   ├── pc_fci_discovery.py  # PC + FCI (causallearn)
│   ├── lingam_discovery.py  # DirectLiNGAM
│   ├── notears_discovery.py # NOTEARS-MLP
│   ├── granger_population.py
│   ├── ensemble_dag.py      # Bootstrap-frequency consensus DAG
│   └── trimester_comparison.py
├── cascade_analysis/        # Causal cascade extraction & scoring
│   ├── cascade_identification.py
│   ├── outcome_specific_cascades.py
│   └── pathway_scoring.py
├── counterfactual/          # Causal effect / intervention-window estimation
│   ├── dowhy_estimation.py  # PSW-ATT (DoWhy)
│   ├── iv_analysis.py        # IV-2SLS (linearmodels)
│   ├── rdd_analysis.py        # RDD (rdrobust)
│   ├── window_quantification.py
│   └── sensitivity_analysis.py
├── validation/              # External replication
│   ├── cross_center_replication.py
│   └── guideline_comparison.py
├── visualization/           # Plotting utilities (DAGs, cascades, heatmaps)
├── pipeline/
│   ├── run_all.py           # End-to-end orchestrator (Phase 1 → 6)
│   └── run_robustness.py
├── data/
│   └── example/
│       ├── generate_example_data.py  # Generates a synthetic cohort
│       └── example_cohort.parquet    # 500-record synthetic dataset
├── results/                 # Runtime output directory (created when you run the pipeline)
└── requirements.txt
```

---

## Installation

```bash
git clone https://github.com/yangfangs/pregnancy-causal-discovery.git
cd pregnancy-causal-discovery

conda create -n causal python=3.10 -y
conda activate causal
pip install -r requirements.txt
```

Python 3.10+. The causal-discovery backends (`causallearn`, `lingam`, `dowhy`,
`linearmodels`, `rdrobust`) are required to run the full pipeline; versions are pinned in
`requirements.txt`.

---

## Quick Start — run the framework on synthetic data

No patient data is required. The generator produces both a synthetic cohort and the
harmonized-layout files the pipeline consumes, so the orchestrator runs straight through:

```bash
# 1. Generate the synthetic dataset (writes data/example/ + results/harmonized/)
python data/example/generate_example_data.py

# 2. Run the full framework end-to-end on the synthetic data
python pipeline/run_all.py
```

`run_all.py` detects the harmonized files produced in step 1 and proceeds through causal
discovery → cascade analysis → counterfactual estimation → visualization. Outputs are
written under `results/`.

> The synthetic dataset reproduces the column schema and approximate marginal distributions
> of the real cohort (with planted associations for illustration) but contains **no real
> patient information**. It is intended for exercising and adapting the code, not for
> drawing clinical conclusions.

---

## Running on your own data

To apply the framework to a real cohort, provide cleaned per-center laboratory panels and
point the pipeline at them:

```bash
export CAUSAL_DATA_ROOT=/path/to/your/cleaned_data
python pipeline/run_all.py
```

Expected layout:
`{CAUSAL_DATA_ROOT}/{CENTER}/{BloodRoutine,Biochemistry,Coagulation,UrineRoutine}_cleaned.parquet`.
Center names, column-name synonym mappings, outcome label mappings, and trimester bounds are
configured in [config/settings.py](config/settings.py). Phase 1 (`data_prep/harmonize_local.py`)
harmonizes and standardizes these into `results/harmonized/`, which the rest of the pipeline
consumes.

---

## Methods & Key Parameters

| Phase | Method | Key parameters |
|---|---|---|
| Feature selection | Variance + coverage filter | `MIN_FEATURE_COVERAGE`, `MIN_STRATUM_SIZE` |
| Causal discovery | PC + FCI + DirectLiNGAM + NOTEARS-MLP | `BOOTSTRAP_N`, `PC_ALPHA_SWEEP` |
| Ensemble DAG | Weighted bootstrap-frequency aggregation | `ENSEMBLE_THRESHOLD` |
| Counterfactual | PSW-ATT (DoWhy), IV-2SLS, RDD | `GW_WINDOWS`, `TREATMENT_THRESHOLDS` |
| Sensitivity | Age subgroup, imputation M-sweep, threshold sweep | see `config/settings.py` |

All tunable parameters are defined in [config/settings.py](config/settings.py).

---

## Scope

This repository is released as a **methods framework**. To keep the release focused and to
respect pre-publication and data-governance constraints, it intentionally does **not**
include:

- the study's pre-computed results (consensus graphs, cascades, counterfactual estimates),
- manuscript-specific figure/table generation code, or
- the manuscript and any patient-level data.

Everything needed to run the methods on your own (or the synthetic) data is included.

---

## License

Released under the [MIT License](LICENSE).

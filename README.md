# Analysis Code for "Gut Microbiome and Body Symptoms Covary with Lifestyle"

Unno T, Choi JB, Lee SH, You Y, Na CS. Gut Microbiome and Body Symptoms Covary with Lifestyle: Implications for Microbiome-Informed Health Management. *mSystems*. 2026.

## Overview

Python and R code for analyzing 16S rRNA gut microbiome data and health questionnaire responses from 774 Korean adults in the Dongshin University Korean Medicine Examination (KME) cohort.

## Data availability

- **16S rRNA sequences**: NCBI BioProject [PRJNA1472377](https://www.ncbi.nlm.nih.gov/bioproject/PRJNA1472377)
- **Questionnaire data**: Available from the Korea Institute of Oriental Medicine (KIOM) upon reasonable request

## Requirements

### Python (≥ 3.10)
numpy, pandas, scipy, scikit-learn, xgboost, statsmodels, matplotlib

### R (≥ 4.2)
ALDEx2, DESeq2, ANCOM-BC2, LinDA, MaAsLin2 (for multi-tool differential abundance)

## Code structure

| File | Description |
|---|---|
| `config.py` | Paths, outcome definitions, questionnaire block definitions |
| `data_join.py` | Data loading: ASV table → genus-level relative abundance, CLR transform, KMEDATA loading |
| `features.py` | Questionnaire preprocessing, block assignment, feature matrix construction (M1/M2/M3) |
| `model.py` | ML model evaluation (XGBoost, logistic regression, random forest), DeLong test, SHAP |
| `run.py` | Main ML pipeline: data join → feature engineering → M1/M2/M3 model comparison |
| `run_algorithm_comparison.py` | Three-classifier comparison (XGBoost, LR, RF) across all outcomes |
| `analysis.py` | Differential abundance, diversity, cold-heat constitution, enterotype analyses |
| `run_age_sex_adjusted.py` | Symptom-microbiome associations with age+sex covariates (OLS, Type II F-test) |
| `run_da_comparison.py` | Multi-tool DA consensus (calls R scripts for ALDEx2, DESeq2, ANCOM-BC2, LinDA, MaAsLin2) |
| `run_pathway_aldex2.py` | PICRUSt2 pathway differential abundance |
| `mechanisms.py` | HAM/DAM ecological module analysis |
| `paper2_figures.py` | KM ablation test, block importance (Fig. 1), cold-heat analysis |
| `make_figures.py` | Main manuscript figures (Fig. 1–5, Fig. S1–S5) |
| `make_supp_tables.py` | Supplementary tables (Table S1–S6) |
| `r_scripts/` | R scripts for differential abundance tools |

## Usage

```bash
# Set up (adjust paths in config.py first)
cd /path/to/microbiome_report

# Run ML model comparison (Table 2)
python -m ml.run --source dada2

# Run all analyses (DA, diversity, enterotypes, cold-heat)
python -m ml.analysis

# Run age+sex adjusted symptom-microbiome associations
python -m ml.run_age_sex_adjusted

# Run multi-tool DA comparison
python -m ml.run_da_comparison

# Generate figures
python -m ml.make_figures
```

## Key analyses

1. **ML model comparison** (`run.py`): Questionnaire-only (M1) vs microbiome-only (M2) vs combined (M3) prediction of 10 health outcomes using 5-fold stratified CV.
2. **Ablation analysis** (`paper2_figures.py`): Effect of removing 64 body constitution features on prediction.
3. **Differential abundance** (`run_da_comparison.py`, `analysis.py`): Four-tool consensus (OLS, MaAsLin2, LinDA, ANCOM-BC2) with age+sex covariates identifying 20 genus-outcome associations.
4. **Symptom-microbiome associations** (`run_age_sex_adjusted.py`): Age+sex adjusted OLS for 51 body constitution items × 147 genera.
5. **Ecological modules** (`mechanisms.py`): HAM/DAM module balance scores, residualized for age+sex.
6. **Enterotype analysis** (`analysis.py`): PAM clustering on Jensen-Shannon divergence.

## License

This code is provided for reproducibility of the published results. Please cite the paper if you use this code.

# Analysis Code for "Gut Microbiome and Body Symptoms Covary with Lifestyle"

Unno T, Choi JB, Lee SH, You Y, Na CS. Gut Microbiome and Body Symptoms Covary with Lifestyle: Implications for Microbiome-Informed Health Management. *mSystems*. 2026.

## Overview

Python and R code for analyzing 16S rRNA gut microbiome data and health questionnaire responses from 774 Korean adults in the Dongshin University Korean Medicine Examination (KME) cohort.

## Data availability

- **16S rRNA sequences**: NCBI BioProject [PRJNA1472377](https://www.ncbi.nlm.nih.gov/bioproject/PRJNA1472377)
- **Questionnaire data**: Available from the Korea Institute of Oriental Medicine (KIOM) upon reasonable request

## Requirements

### Python (≥ 3.10)
numpy, pandas, scipy, scikit-learn, xgboost, statsmodels, matplotlib; scikit-bio (longevity Table 2)

### R (≥ 4.2)
ALDEx2, DESeq2, ANCOM-BC2, LinDA, MaAsLin2 (for multi-tool differential abundance); ANCOMBC ≥ 2.8, phyloseq, optparse, jsonlite (longevity ANCOM-BC2)

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

## Longevity-village analysis

Code for the companion manuscript "Host-Adapted Lactobacilli and Fiber-Degrader Depletion in the Gut Microbiome of Korean Longevity-Village Elders" (in preparation). It compares 1,256 elderly residents of ten designated longevity villages in Jeollanam-do (rural cohort, RC; "Muan"/"MB" in the code) with the 828-adult DS health-checkup cohort. The primary comparison is RC vs DS adults aged 60+ (n = 192).

### Data

- **16S rRNA sequences**: NCBI BioProjects [PRJNA1472377](https://www.ncbi.nlm.nih.gov/bioproject/PRJNA1472377) (DS) and [PRJNA1476885](https://www.ncbi.nlm.nih.gov/bioproject/PRJNA1476885) (RC), processed together with 16S-Pipeline (2,095 samples, 2,084 after QC)
- **RC metadata and questionnaires**: available upon reasonable request

Set the paths in `ml/longevity_config.py`. The defaults expect:

```
data/
├── kmedata_merged.csv          # DS questionnaire (as for the lifestyle analysis)
├── dada2_combined/             # asv_table.tsv, taxonomy.tsv for the combined DS + RC run
└── longevity/
    ├── MB.txt                  # RC metadata: run_year, Sex, Height, Weight, Sample, Birthdate
    ├── sample_groups.csv       # sample_id, cohort, village_type, region (RC sub-district)
    └── questionnaires/         # per-wave RC questionnaires: 22.xlsx … 25.xlsx
```

`data/` and `ml/output/` are git-ignored. Outputs are written to `ml/output/longevity/`.

### Scripts

| File | Description |
|---|---|
| `longevity_config.py` | Paths for the combined DS + RC data and outputs |
| `longevity_muan.py` | Data loading (DS + RC; ages 50–110), cohort summary (Table 1 values), Bray-Curtis PERMANOVA (Table 2 F and P; PERMDISP not included), alpha diversity (Fig. S2), RC vs DS 60+ differential abundance statistics, enterotypes (Fig. S3), PCoA (Fig. S1) |
| `longevity_table_s1.py` | Table S1: 153 genera at ≥10% prevalence, CLR (pseudocount 1e-6), Mann-Whitney with BH-FDR, Cliff's delta, prevalence by group |
| `longevity_fig3.py` | Fig. 3: CLR box plots of 14 key genera across DS young, DS 60+ and RC (5 × 3 grid) |
| `longevity_sensitivity.py` | Table S5 sensitivity analyses on the fixed 153-genus set: (i) ages 60–69 with PERMANOVA, (iii) sex-stratified tests and cohort × sex interaction for the RC-enriched key genera, (iv) shared sampling years 2022 + 2024, (v) rarefaction to 10,000 genus-classified reads, (vi) exclusion of the non-rural sites |
| `longevity_table2.py` | Table 2: Bray-Curtis PERMANOVA (F, R², P; 999 permutations) for cohort, sex, age group and the restricted cohort comparisons, plus PERMDISP (distance to spatial median) for the three cohort comparisons; requires scikit-bio |
| `longevity_fiber_index.py` | Composite fiber-degrader index (summed relative abundance of 10 genera; 7-genus version without the DS-60+ outlier genera): RC vs DS 60+ and in the 60–69 subset (Mann-Whitney, Cohen's d), plus within-RC Spearman correlations with summed lactobacilli |
| `longevity_ancombc2.py` | (ii) ANCOM-BC2 with sex as covariate (Table S3) and its agreement with the Mann-Whitney results; calls `r_scripts/run_ancombc2_longevity.R` |
| `longevity_extra.py` | Fig. S4 heatmap (plots precomputed diet–microbiome correlations from `tableS4_diet_microbiome.csv`) and additional exploratory analyses (microbiome-age model, age-genus overlap, lifestyle vs age gap) |
| `longevity_graphical_abstract.py` | Graphical abstract (values typed in from the results) |
| `longevity_picrust2.py` | Fig. S5 / PICRUSt2 results: DS 60–69 vs RC 60–69 on the unrarefied and rarefied (10,000 reads) runs — NSTI, butyrate-synthesis KOs by route and summed, KO and pathway Mann-Whitney with BH-FDR, key pathways, and the three-group (DS young / DS 60–69 / RC 60–69) pathway classification; `--make-rarefied-input` writes the rarefied PICRUSt2 input |
| `run_picrust2_longevity.sh` | PICRUSt2 v2.6.3 on the unrarefied combined ASV table, with an exploratory KO/pathway summary |

### Usage

```bash
cd /path/to/kme-microbiome-analysis

# Cohort summary, diversity, PERMANOVA, enterotypes, DA statistics
python -m ml.longevity_muan

# Table S1 and Fig. 3
python -m ml.longevity_table_s1
python -m ml.longevity_fig3

# Sensitivity analyses (Table S5) and ANCOM-BC2 (Table S3; requires R + ANCOMBC)
python -m ml.longevity_table2
python -m ml.longevity_sensitivity
python -m ml.longevity_fiber_index
python -m ml.longevity_ancombc2 --threads 8

# Fig. S4 and exploratory analyses
python -m ml.longevity_extra

# Graphical abstract
python -m ml.longevity_graphical_abstract

# PICRUSt2 (requires a conda env named "picrust2")
bash ml/run_picrust2_longevity.sh                 # unrarefied run → ml/output/longevity/picrust2_out
python -m ml.longevity_picrust2 --make-rarefied-input
#   then run picrust2_pipeline.py on the printed files → ml/output/longevity/picrust2_out_rarefied
python -m ml.longevity_picrust2
```

`data_join.load_genus_abundance()` now takes optional `asv_table` and `taxonomy` arguments. The defaults are unchanged, so the lifestyle analysis above behaves exactly as before.

## License

This code is provided for reproducibility of the published results. Please cite the paper if you use this code.

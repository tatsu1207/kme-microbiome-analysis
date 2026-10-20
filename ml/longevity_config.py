"""
Longevity-village analysis configuration (RC vs DS)
Paths for the combined DS + rural-cohort data — adjust these to your environment.
"""
from ml.config import PROJECT_DIR

# === Paths ===
# DADA2 output from 16S-Pipeline for the combined run: DS (PRJNA1472377) +
# rural cohort, RC (PRJNA1476885); 2,095 samples, 2,084 after QC
LONGEVITY_DADA2_DIR = PROJECT_DIR / "data/dada2_combined"
LONGEVITY_ASV_TABLE = LONGEVITY_DADA2_DIR / "asv_table.tsv"
LONGEVITY_TAXONOMY = LONGEVITY_DADA2_DIR / "taxonomy.tsv"

# RC participant metadata (available upon reasonable request)
# MB.txt: tab-separated; columns run_year, Sex, Height, Weight, Sample, Birthdate (YYYYMMDD)
MB_META = PROJECT_DIR / "data/longevity/MB.txt"
# MB.xlsx: 2022-wave roster used only for optional repeat-participant matching
MB_NAMES = PROJECT_DIR / "data/longevity/MB.xlsx"
# Per-wave RC questionnaires (22.xlsx, 23.xlsx, 24.xlsx, 25.xlsx)
MB_QUESTIONNAIRE_DIR = PROJECT_DIR / "data/longevity/questionnaires"

# Outputs (figures, tables)
LONGEVITY_OUT = PROJECT_DIR / "ml/output/longevity"

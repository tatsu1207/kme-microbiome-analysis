"""
Step 1: Join KMEDATA questionnaire with DADA2 microbiome data.

Produces a single merged DataFrame keyed on subject ID with:
  - Questionnaire features (demographics, lifestyle, labs, KM, ...)
  - Genus-level relative abundances (CLR-transformed)
  - PICRUSt2 pathway abundances
  - Outcome variables (metabolic health indicators)

Usage:
    python -m ml.data_join          # writes output/joined.pkl
"""
import gzip
import sys

import numpy as np
import pandas as pd

from ml.config import (
    ASV_TABLE, CLINIC_INSTRUMENT_COLS, DADA2_DIR, KMEDATA_CSV, MIN_AVAILABILITY,
    MIN_ASV_PREVALENCE, OUTCOME_LEAKAGE_COLS, OUTPUT_DIR, PICRUST2_PATHWAYS,
    TAXONOMY, TOP_N_ASVS, TOP_N_GENERA, TOP_N_PATHWAYS,
)


# ---------------------------------------------------------------------------
# 1. Load & normalize KMEDATA
# ---------------------------------------------------------------------------
def load_kmedata() -> pd.DataFrame:
    """Load KMEDATA and normalize subject IDs (- → _)."""
    df = pd.read_csv(KMEDATA_CSV, encoding="utf-8-sig", dtype=str)
    df["sample_id"] = df["USUBJID"].str.replace("-", "_", regex=False)
    df = df.set_index("sample_id")
    return df


# ---------------------------------------------------------------------------
# 2. Load DADA2 ASV table → genus-level relative abundance
# ---------------------------------------------------------------------------
def load_genus_abundance(asv_table=ASV_TABLE, taxonomy=TAXONOMY) -> pd.DataFrame:
    """
    Collapse ASV table to genus level and compute relative abundances.
    asv_table/taxonomy default to config paths; the longevity analysis passes
    the combined DS + RC run (see longevity_config.py).
    Returns: DataFrame (samples × genera), values = relative abundance [0,1].
    """
    # ASV count table: rows = ASVs, cols = samples
    asv = pd.read_csv(asv_table, sep="\t", index_col=0)
    # Drop non-sample columns (e.g. 'sequence' from combined pipeline)
    non_sample_cols = [c for c in asv.columns if c in ("sequence",)]
    if non_sample_cols:
        asv = asv.drop(columns=non_sample_cols)

    # Taxonomy mapping: ASV → Genus
    tax = pd.read_csv(taxonomy, sep="\t", index_col=0)
    tax["genus_label"] = tax["Genus"].fillna("Unclassified")

    # Keep only ASVs present in taxonomy
    common_asvs = asv.index.intersection(tax.index)
    asv = asv.loc[common_asvs]
    tax = tax.loc[common_asvs]

    # Collapse to genus level (sum counts per genus per sample)
    asv["genus_label"] = tax["genus_label"]
    genus_counts = asv.groupby("genus_label").sum()

    # Relative abundance (per sample, columns sum to 1)
    rel_abund = genus_counts.div(genus_counts.sum(axis=0), axis=1)

    # Transpose to samples × genera
    result = rel_abund.T

    # Drop samples with zero total reads (failed sequencing)
    zero_mask = result.sum(axis=1) == 0
    if zero_mask.any():
        result = result[~zero_mask]

    return result


def clr_transform(df: pd.DataFrame) -> pd.DataFrame:
    """Centered log-ratio transform. Adds pseudocount to handle zeros."""
    pseudocount = 1e-6
    log_data = np.log(df + pseudocount)
    geometric_mean = log_data.mean(axis=1)
    clr = log_data.sub(geometric_mean, axis=0)
    return clr


def select_top_genera(genus_df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Select top N genera by variance across samples."""
    variances = genus_df.var(axis=0).sort_values(ascending=False)
    # Exclude "Unclassified" from top selection
    variances = variances[variances.index != "Unclassified"]
    top = variances.head(n).index.tolist()
    return genus_df[top]


# ---------------------------------------------------------------------------
# 2b. Load DADA2 ASV table directly (no genus collapse)
# ---------------------------------------------------------------------------
def load_asv_abundance() -> pd.DataFrame:
    """
    Load ASV count table, filter by prevalence, compute relative abundance.
    Returns: DataFrame (samples × ASVs), values = relative abundance [0,1].
    """
    asv = pd.read_csv(ASV_TABLE, sep="\t", index_col=0)

    # Filter by prevalence: keep ASVs present in >= MIN_ASV_PREVALENCE of samples
    prevalence = (asv > 0).mean(axis=1)
    asv = asv.loc[prevalence >= MIN_ASV_PREVALENCE]

    # Relative abundance (per sample, columns sum to 1)
    rel_abund = asv.div(asv.sum(axis=0), axis=1)

    # Transpose to samples × ASVs
    return rel_abund.T


def select_top_features(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Select top N features by variance across samples."""
    variances = df.var(axis=0).sort_values(ascending=False)
    top = variances.head(n).index.tolist()
    return df[top]


# ---------------------------------------------------------------------------
# 3. Load PICRUSt2 pathway abundances
# ---------------------------------------------------------------------------
def load_pathways() -> pd.DataFrame:
    """Load PICRUSt2 MetaCyc pathway abundances (samples × pathways)."""
    if not PICRUST2_PATHWAYS.exists():
        print(f"  Warning: PICRUSt2 pathways not found at {PICRUST2_PATHWAYS}")
        return pd.DataFrame()
    with gzip.open(PICRUST2_PATHWAYS, "rt") as f:
        df = pd.read_csv(f, sep="\t", index_col=0)
    # Transpose: rows=pathways, cols=samples → samples × pathways
    return df.T


def select_top_pathways(pw_df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Select top N pathways by variance."""
    variances = pw_df.var(axis=0).sort_values(ascending=False)
    top = variances.head(n).index.tolist()
    return pw_df[top]


# ---------------------------------------------------------------------------
# 4. Define outcome variables
# ---------------------------------------------------------------------------
def compute_outcomes(qdf: pd.DataFrame) -> pd.DataFrame:
    """Compute binary outcomes: metabolic + digestive health."""
    out = pd.DataFrame(index=qdf.index)

    # Helper: safe numeric conversion
    def to_num(col_name):
        return pd.to_numeric(qdf.get(col_name, pd.Series(dtype=float)), errors="coerce")

    sex = qdf.get("SEX", pd.Series(dtype=str))

    # --- Metabolic outcomes (defined from lab values) ---

    # Obesity: PBF >= 25% (M) / >= 30% (F)
    pbf = to_num("PBF")
    out["y_obesity"] = np.where(
        sex == "M", (pbf >= 25).astype(float), (pbf >= 30).astype(float)
    )
    out.loc[pbf.isna(), "y_obesity"] = np.nan

    # Hypertension: SBP >= 130 or DBP >= 85
    sbp = to_num("SYSBP")
    dbp = to_num("DIABP")
    out["y_hypertension"] = ((sbp >= 130) | (dbp >= 85)).astype(float)
    out.loc[sbp.isna() & dbp.isna(), "y_hypertension"] = np.nan

    # Pre-diabetes: fasting glucose >= 100
    gluc = to_num("GLUC")
    out["y_prediabetes"] = (gluc >= 100).astype(float)
    out.loc[gluc.isna(), "y_prediabetes"] = np.nan

    # Liver dysfunction: ALT > 40 or GGT > 60
    alt = to_num("ALT")
    ggt = to_num("GGT")
    out["y_liver"] = ((alt > 40) | (ggt > 60)).astype(float)
    out.loc[alt.isna() & ggt.isna(), "y_liver"] = np.nan

    # Multi-morbidity count (0-4)
    y_cols = ["y_obesity", "y_hypertension", "y_prediabetes", "y_liver"]
    out["y_multimorbidity"] = out[y_cols].sum(axis=1)
    out.loc[out[y_cols].isna().any(axis=1), "y_multimorbidity"] = np.nan

    # --- Digestive health outcomes (defined from self-report) ---

    # 1. Digestive symptoms: any symptoms present (DIGEST02_01: 1=no symptoms, 2=has)
    d01 = qdf.get("DIGEST02_01", pd.Series(dtype=str))
    out["y_digestive_symptoms"] = (d01 == "2").astype(float)
    out.loc[~d01.isin(["1", "2"]), "y_digestive_symptoms"] = np.nan

    # 2. Multiple digestive symptoms (>=2 of: bloating, distension, early satiety,
    #    overeating, heartburn, epigastric pain, nausea, post-meal fatigue)
    #    Coding: 1=has symptom, 2=doesn't
    symptom_cols = [f"DIGEST02_{i:02d}" for i in range(2, 10)]
    symptom_count = pd.DataFrame(index=qdf.index)
    for col in symptom_cols:
        s = qdf.get(col, pd.Series(dtype=str))
        symptom_count[col] = (s == "1").astype(float)
        symptom_count.loc[~s.isin(["1", "2"]), col] = np.nan
    total_symptoms = symptom_count.sum(axis=1)
    out["y_multi_digestive"] = (total_symptoms >= 2).astype(float)
    out.loc[symptom_count.isna().any(axis=1), "y_multi_digestive"] = np.nan

    # 3. Abnormal stool form: Bristol Stool Scale 1-2 (constipation) or 6-7 (diarrhea)
    bss = to_num("STOOL03")
    out["y_abnormal_stool"] = ((bss <= 2) | (bss >= 6)).astype(float)
    out.loc[bss.isna(), "y_abnormal_stool"] = np.nan

    # 4. Bowel difficulty: frequent straining (STOOL06 >= 3) or
    #    frequent incomplete evacuation (STOOL07 >= 3)
    strain = to_num("STOOL06")
    residual = to_num("STOOL07")
    out["y_bowel_difficulty"] = ((strain >= 3) | (residual >= 3)).astype(float)
    out.loc[strain.isna() & residual.isna(), "y_bowel_difficulty"] = np.nan

    # 5. Poor gut health composite: any of symptoms + abnormal stool + bowel difficulty
    gut_cols = ["y_digestive_symptoms", "y_abnormal_stool", "y_bowel_difficulty"]
    gut_sum = out[gut_cols].sum(axis=1)
    out["y_poor_gut"] = (gut_sum >= 2).astype(float)
    out.loc[out[gut_cols].isna().any(axis=1), "y_poor_gut"] = np.nan

    return out


# ---------------------------------------------------------------------------
# 5. Filter questionnaire columns by availability
# ---------------------------------------------------------------------------
def filter_questionnaire_cols(qdf: pd.DataFrame, min_rate: float) -> list[str]:
    """Return columns with non-empty rate >= min_rate."""
    # Exclude ID/meta columns AND outcome-defining lab columns (leakage prevention)
    skip = {"USUBJID", "YEAR", "SITE", "AGEGROUP", "sample_id", "SUBINIT",
            "BIRTHDTC", "BIRTH_YEAR", "BIRTH_YEAR_ESTIMATED",
            "GMEDTC", "MHSTDTC", "MHMFSTDTC", "TESTDATE"}
    skip |= OUTCOME_LEAKAGE_COLS
    skip |= CLINIC_INSTRUMENT_COLS
    candidates = [c for c in qdf.columns if c not in skip]

    good = []
    for col in candidates:
        rate = qdf[col].replace("", np.nan).dropna().shape[0] / len(qdf)
        if rate >= min_rate:
            good.append(col)
    return good


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def build_joined_dataset() -> dict[str, pd.DataFrame]:
    """
    Returns dict with:
      - "questionnaire": filtered questionnaire features (n × q)
      - "genus_clr": CLR-transformed top genera (n × g)
      - "pathways": top PICRUSt2 pathways (n × p)
      - "outcomes": binary outcomes + multi-morbidity (n × 5)
      - "meta": YEAR, SEX, AGE for stratification
    All aligned to same subject index (paired DS subjects only).
    """
    print("Loading KMEDATA questionnaire...")
    kme = load_kmedata()

    print("Loading DADA2 genus abundances...")
    genus_rel = load_genus_abundance()

    # Identify paired subjects
    paired_ids = kme.index.intersection(genus_rel.index)
    print(f"Paired subjects: {len(paired_ids)}")

    # Subset to paired
    kme = kme.loc[paired_ids]
    genus_rel = genus_rel.loc[paired_ids]

    # Filter DS subjects only (questionnaire is KMEDATA = DS)
    ds_mask = kme.index.str.contains("_DS_")
    kme = kme.loc[ds_mask]
    genus_rel = genus_rel.loc[ds_mask]
    paired_ids = kme.index
    print(f"DS paired subjects: {len(paired_ids)}")

    # Genus: select top N, CLR-transform
    print(f"Selecting top {TOP_N_GENERA} genera by variance...")
    genus_top = select_top_genera(genus_rel, TOP_N_GENERA)
    genus_clr = clr_transform(genus_top)
    genus_clr.columns = ["genus__" + c for c in genus_clr.columns]

    # PICRUSt2 pathways
    print("Loading PICRUSt2 pathways...")
    pw = load_pathways()
    if len(pw) > 0:
        pw = pw.loc[pw.index.intersection(paired_ids)]
        print(f"Selecting top {TOP_N_PATHWAYS} pathways by variance...")
        pw_top = select_top_pathways(pw, TOP_N_PATHWAYS)
        pw_clr = clr_transform(pw_top)
        pw_clr.columns = ["pathway__" + c for c in pw_clr.columns]
    else:
        pw_clr = pd.DataFrame(index=paired_ids)

    # Questionnaire: filter by availability
    print("Filtering questionnaire columns...")
    good_cols = filter_questionnaire_cols(kme, MIN_AVAILABILITY)
    print(f"Questionnaire columns with >={MIN_AVAILABILITY*100:.0f}% data: {len(good_cols)}")
    quest = kme[good_cols].copy()

    # Outcomes
    print("Computing outcome variables...")
    outcomes = compute_outcomes(kme)

    # Meta for stratification
    meta = kme[["YEAR", "SEX", "AGE"]].copy()
    meta["AGE"] = pd.to_numeric(meta["AGE"], errors="coerce")

    result = {
        "questionnaire": quest,
        "genus_clr": genus_clr,
        "pathways": pw_clr,
        "outcomes": outcomes,
        "meta": meta,
    }

    # Print summary
    print("\n=== Dataset Summary ===")
    print(f"Subjects:           {len(paired_ids)}")
    print(f"Questionnaire cols: {quest.shape[1]}")
    print(f"Genus features:     {genus_clr.shape[1]}")
    print(f"Pathway features:   {pw_clr.shape[1]}")
    print(f"Outcome prevalence:")
    for col in outcomes.columns:
        valid = outcomes[col].dropna()
        if col == "y_multimorbidity":
            print(f"  {col}: mean={valid.mean():.2f}, distribution={valid.value_counts().sort_index().to_dict()}")
        else:
            print(f"  {col}: {valid.sum():.0f}/{len(valid)} ({valid.mean()*100:.1f}%)")

    return result


def save_joined(data: dict[str, pd.DataFrame]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "joined.pkl"
    pd.to_pickle(data, out_path)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    data = build_joined_dataset()
    save_joined(data)

"""
Step 1 (mothur variant): Join KMEDATA questionnaire with mothur OTU-based microbiome data.

Loads mothur .shared + .taxonomy files, collapses OTUs to genus level,
and produces the same output format as data_join.py (DADA2 variant).

Note: mothur pipeline has no PICRUSt2 pathways, so M2 = genus CLR only.
"""
import numpy as np
import pandas as pd

from ml.config import (
    CLINIC_INSTRUMENT_COLS, KMEDATA_CSV, MIN_AVAILABILITY,
    MOTHUR_SHARED, MOTHUR_TAXONOMY,
    OUTCOME_LEAKAGE_COLS, OUTPUT_DIR, TOP_N_GENERA,
)
from ml.data_join import (
    clr_transform,
    compute_outcomes,
    filter_questionnaire_cols,
    load_kmedata,
    select_top_genera,
)


def load_mothur_genus_abundance() -> pd.DataFrame:
    """
    Load mothur .shared file and .taxonomy, collapse OTUs to genus level.
    Returns: DataFrame (samples x genera), values = relative abundance [0,1].
    """
    # Load OTU count table: rows = samples, first 3 cols = label, Group, numPhylos
    shared = pd.read_csv(MOTHUR_SHARED, sep="\t")
    sample_ids = shared["Group"]
    otu_cols = [c for c in shared.columns if c.startswith("Phylo")]
    counts = shared[otu_cols].copy()
    counts.index = sample_ids

    # Load taxonomy mapping: OTU -> genus
    tax = pd.read_csv(MOTHUR_TAXONOMY, sep="\t")
    tax = tax.set_index("OTU")

    # Parse genus from semicolon-delimited taxonomy string
    def extract_genus(tax_str):
        parts = [p.strip() for p in tax_str.rstrip(";").split(";") if p.strip()]
        if len(parts) >= 6:
            return parts[5]
        return "Unclassified"

    tax["genus_label"] = tax["Taxonomy"].apply(extract_genus)

    # Build OTU -> genus mapping for OTUs present in shared file
    common_otus = [c for c in otu_cols if c in tax.index]
    otu_genus = tax.loc[common_otus, "genus_label"]

    # Collapse to genus level: sum counts per genus per sample
    genus_counts = pd.DataFrame(index=counts.index)
    for genus in otu_genus.unique():
        otus_in_genus = otu_genus[otu_genus == genus].index.tolist()
        genus_counts[genus] = counts[otus_in_genus].sum(axis=1)

    # Relative abundance (per sample, rows sum to 1)
    row_sums = genus_counts.sum(axis=1)
    rel_abund = genus_counts.div(row_sums, axis=0)

    return rel_abund


def build_joined_dataset_mothur() -> dict[str, pd.DataFrame]:
    """
    Build joined dataset using mothur OTU data instead of DADA2.
    Returns same dict structure as data_join.build_joined_dataset(),
    but with no 'pathways' key (mothur has no PICRUSt2).
    """
    print("Loading KMEDATA questionnaire...")
    kme = load_kmedata()

    print("Loading mothur genus abundances...")
    genus_rel = load_mothur_genus_abundance()

    # Normalize sample IDs to match KMEDATA format
    # mothur uses 20_DS_0036, kmedata uses 20_DS_0036 (after - to _ conversion)
    paired_ids = kme.index.intersection(genus_rel.index)
    print(f"Paired subjects: {len(paired_ids)}")

    kme = kme.loc[paired_ids]
    genus_rel = genus_rel.loc[paired_ids]

    # Filter DS subjects only
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

    # Questionnaire: filter by availability
    print("Filtering questionnaire columns...")
    good_cols = filter_questionnaire_cols(kme, MIN_AVAILABILITY)
    print(f"Questionnaire columns with >={MIN_AVAILABILITY*100:.0f}% data: {len(good_cols)}")
    quest = kme[good_cols].copy()

    # Outcomes
    print("Computing outcome variables...")
    outcomes = compute_outcomes(kme)

    # Meta
    meta = kme[["YEAR", "SEX", "AGE"]].copy()
    meta["AGE"] = pd.to_numeric(meta["AGE"], errors="coerce")

    result = {
        "questionnaire": quest,
        "genus_clr": genus_clr,
        "outcomes": outcomes,
        "meta": meta,
    }

    # Print summary
    print("\n=== Dataset Summary (Mothur) ===")
    print(f"Subjects:           {len(paired_ids)}")
    print(f"Questionnaire cols: {quest.shape[1]}")
    print(f"Genus features:     {genus_clr.shape[1]}")
    print(f"Outcome prevalence:")
    for col in outcomes.columns:
        valid = outcomes[col].dropna()
        if col == "y_multimorbidity":
            print(f"  {col}: mean={valid.mean():.2f}, distribution={valid.value_counts().sort_index().to_dict()}")
        else:
            print(f"  {col}: {valid.sum():.0f}/{len(valid)} ({valid.mean()*100:.1f}%)")

    return result

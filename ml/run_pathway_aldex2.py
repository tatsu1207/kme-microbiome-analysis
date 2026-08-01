"""
Run ALDEx2 on PICRUSt2 pathway and KO data for each outcome.

Usage:
    cd /path/to/kme-microbiome-analysis
    python -m ml.run_pathway_aldex2
"""
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from ml.config import DADA2_DIR, KMEDATA_CSV
from ml.data_join import load_kmedata, load_pathways, compute_outcomes

R_SCRIPT = Path(__file__).resolve().parent / "r_scripts" / "run_aldex2.R"
CONDA_ENV = "analysis_16S"
OUTPUT_DIR = Path("ml/output/analysis")

METABOLIC = ["y_hypertension", "y_prediabetes", "y_liver", "y_obesity"]
ALL_OUTCOMES = METABOLIC + [
    "y_digestive_symptoms", "y_multi_digestive", "y_abnormal_stool",
    "y_bowel_difficulty", "y_poor_gut",
]


def run_aldex2(counts_df, meta_df, outcome_col, tmpdir):
    """Run ALDEx2 on a count matrix for one outcome."""
    y = meta_df[outcome_col].dropna()
    common = counts_df.columns.intersection(y.index)
    y = y.loc[common]
    counts = counts_df[common]

    # Remove zero-sum features
    counts = counts[counts.sum(axis=1) > 0]

    # Build metadata
    meta = pd.DataFrame({
        "SampleID": common,
        "group": y.astype(int).map({0: "healthy", 1: "disease"}).values,
    })

    n_disease = (meta["group"] == "disease").sum()
    n_healthy = (meta["group"] == "healthy").sum()
    if n_disease < 10 or n_healthy < 10:
        return None

    counts_path = Path(tmpdir) / "counts.tsv"
    meta_path = Path(tmpdir) / "metadata.tsv"
    output_path = Path(tmpdir) / "aldex2_results.tsv"

    counts.to_csv(counts_path, sep="\t")
    meta.to_csv(meta_path, sep="\t", index=False)

    cmd = [
        "conda", "run", "-n", CONDA_ENV,
        "Rscript", str(R_SCRIPT),
        "--counts", str(counts_path),
        "--metadata", str(meta_path),
        "--group_col", "group",
        "--ref_group", "healthy",
        "--test_group", "disease",
        "--output", str(output_path),
        "--threads", "16",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    if result.returncode != 0:
        print(f"      FAILED: {result.stderr[-200:]}")
        return None

    if output_path.exists():
        return pd.read_csv(output_path, sep="\t")
    return None


def main():
    import gzip

    print("Loading data...")
    kme = load_kmedata()
    outcomes = compute_outcomes(kme)
    pw_raw = load_pathways()

    # Get DS cohort
    common = kme.index.intersection(pw_raw.index)
    ds_mask = pd.Series(common).str.contains("_DS_")
    common = common[ds_mask.values]
    outcomes = outcomes.loc[common]
    print(f"DS samples with pathways: {len(common)}")

    # Pathway counts (raw, not relative abundance)
    # PICRUSt2 outputs are already counts (integer-like)
    pw_counts = pw_raw.loc[common].T  # features x samples
    pw_counts = pw_counts.round(0).astype(int)

    # Load pathway descriptions
    pw_desc_path = DADA2_DIR / "picrust2/pathways_out/path_abun_unstrat_described.tsv.gz"
    pw_desc = {}
    if pw_desc_path.exists():
        with gzip.open(pw_desc_path, "rt") as f:
            desc_df = pd.read_csv(f, sep="\t", usecols=["pathway", "description"])
        pw_desc = dict(zip(desc_df["pathway"], desc_df["description"]))

    all_results = []

    for outcome in ALL_OUTCOMES:
        print(f"\n  {outcome}...", end=" ", flush=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            df = run_aldex2(pw_counts, outcomes, outcome, tmpdir)

        if df is not None:
            df["outcome"] = outcome
            df["description"] = df["feature"].map(pw_desc).fillna("")
            n_sig = (df["qvalue"] < 0.05).sum() if "qvalue" in df.columns else 0
            print(f"{len(df)} pathways, {n_sig} significant")
            all_results.append(df)
        else:
            print("skipped")

    if not all_results:
        print("No results!")
        return

    combined = pd.concat(all_results, ignore_index=True)

    # Rename columns to match existing format
    combined = combined.rename(columns={
        "feature": "pathway",
        "log2fc": "log2fc",
        "qvalue": "q_value",
        "pvalue": "p_value",
    })
    combined["significant"] = combined["q_value"] < 0.05
    combined["direction"] = combined["log2fc"].apply(
        lambda x: "up_in_disease" if x > 0 else "down_in_disease"
    )
    # Convert log2fc to CLR-like diff for compatibility
    combined["clr_diff"] = combined["log2fc"]

    combined.to_csv(OUTPUT_DIR / "pathway_enrichment.csv", index=False)
    print(f"\nSaved to {OUTPUT_DIR / 'pathway_enrichment.csv'}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY: Significant pathways (ALDEx2, q < 0.05)")
    print(f"{'='*60}")
    for outcome in ALL_OUTCOMES:
        label = outcome.replace("y_", "").replace("_", " ").title()
        sub = combined[(combined["outcome"] == outcome) & (combined["significant"])]
        print(f"  {label}: {len(sub)} pathways")
        for _, row in sub.head(5).iterrows():
            desc = row["description"][:50] if row["description"] else row["pathway"]
            print(f"    {desc:<55s} log2fc={row['log2fc']:+.3f}  q={row['q_value']:.4f}")


if __name__ == "__main__":
    main()

"""
Run all 5 differential abundance tools from 16S-Pipeline on our dataset.
Tools: ALDEx2, DESeq2, ANCOM-BC2, LinDA, MaAsLin2

Usage:
    cd /home/unnot/github/health-check/microbiome_report
    python -m ml.run_da_comparison
"""
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from ml.config import ASV_TABLE, TAXONOMY, KMEDATA_CSV
from ml.data_join import compute_outcomes, load_kmedata

R_SCRIPTS_DIR = Path("/home/unnot/github/16S-Pipeline/r_scripts")
OUTPUT_DIR = Path("ml/output/da_comparison")
CONDA_ENV = "analysis_16S"

TOOLS = {
    "aldex2": "run_aldex2.R",
    "deseq2": "run_deseq2.R",
    "ancombc": "run_ancombc.R",
    "linda": "run_linda.R",
    "maaslin2": "run_maaslin2.R",
}

METABOLIC_OUTCOMES = ["y_hypertension", "y_prediabetes", "y_liver", "y_obesity"]


def build_genus_counts() -> pd.DataFrame:
    """Build genus-level count matrix (genera × samples) from ASV table."""
    asv = pd.read_csv(ASV_TABLE, sep="\t", index_col=0)
    tax = pd.read_csv(TAXONOMY, sep="\t", index_col=0)
    tax["genus_label"] = tax["Genus"].fillna("Unclassified")

    common = asv.index.intersection(tax.index)
    asv = asv.loc[common]
    asv["genus_label"] = tax.loc[common, "genus_label"]

    genus_counts = asv.groupby("genus_label").sum()
    # Remove Unclassified
    genus_counts = genus_counts.drop("Unclassified", errors="ignore")
    # Convert to int
    genus_counts = genus_counts.round(0).astype(int)
    return genus_counts


def build_outcomes() -> pd.DataFrame:
    """Load KMEDATA and compute binary outcomes."""
    kme = load_kmedata()
    outcomes = compute_outcomes(kme)
    return outcomes


def get_ds_samples(genus_counts, outcomes):
    """Get DS cohort samples present in both microbiome and questionnaire."""
    micro_samples = set(genus_counts.columns)
    quest_samples = set(outcomes.index)
    common = sorted(micro_samples & quest_samples)
    print(f"  Paired samples: {len(common)}")
    return common


def prepare_input_files(genus_counts, outcomes, outcome_col, common_samples, tmpdir):
    """Write count matrix and metadata TSVs for R scripts."""
    y = outcomes[outcome_col].loc[common_samples].dropna()
    samples_with_label = y.index.tolist()

    # Filter to 10%+ prevalence
    counts = genus_counts[samples_with_label]
    prevalence = (counts > 0).mean(axis=1)
    counts = counts[prevalence >= 0.10]

    # Remove zero-sum features
    counts = counts[counts.sum(axis=1) > 0]

    # Metadata
    meta = pd.DataFrame({
        "SampleID": samples_with_label,
        "group": y.astype(int).map({0: "healthy", 1: "disease"}).values,
    })

    counts_path = Path(tmpdir) / "counts.tsv"
    meta_path = Path(tmpdir) / "metadata.tsv"

    counts.to_csv(counts_path, sep="\t")
    meta.to_csv(meta_path, sep="\t", index=False)

    n_disease = (meta["group"] == "disease").sum()
    n_healthy = (meta["group"] == "healthy").sum()
    print(f"    {outcome_col}: {len(samples_with_label)} samples "
          f"({n_disease} disease, {n_healthy} healthy), "
          f"{len(counts)} genera")

    return counts_path, meta_path


def run_r_tool(tool_name, script_name, counts_path, meta_path, output_path, threads=16):
    """Run an R DA script via conda."""
    cmd = [
        "conda", "run", "-n", CONDA_ENV,
        "Rscript", str(R_SCRIPTS_DIR / script_name),
        "--counts", str(counts_path),
        "--metadata", str(meta_path),
        "--group_col", "group",
        "--ref_group", "healthy",
        "--test_group", "disease",
        "--output", str(output_path),
    ]
    if tool_name != "linda":
        cmd.extend(["--threads", str(threads)])

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    if result.returncode != 0:
        print(f"      {tool_name} FAILED: {result.stderr[-200:]}")
        return None

    if output_path.exists():
        df = pd.read_csv(output_path, sep="\t")
        return df
    return None


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Building genus count matrix...")
    genus_counts = build_genus_counts()
    print(f"  {genus_counts.shape[0]} genera × {genus_counts.shape[1]} samples")

    print("Loading outcomes...")
    outcomes = build_outcomes()
    common_samples = get_ds_samples(genus_counts, outcomes)

    all_results = []

    for outcome in METABOLIC_OUTCOMES:
        print(f"\n{'='*60}")
        print(f"Outcome: {outcome}")
        print(f"{'='*60}")

        with tempfile.TemporaryDirectory() as tmpdir:
            counts_path, meta_path = prepare_input_files(
                genus_counts, outcomes, outcome, common_samples, tmpdir
            )

            for tool_name, script_name in TOOLS.items():
                print(f"    Running {tool_name}...", end=" ", flush=True)
                output_path = Path(tmpdir) / f"{tool_name}_results.tsv"

                df = run_r_tool(tool_name, script_name, counts_path, meta_path, output_path)

                if df is not None:
                    df["tool"] = tool_name
                    df["outcome"] = outcome
                    n_sig = (df["qvalue"] < 0.05).sum() if "qvalue" in df.columns else 0
                    print(f"{len(df)} genera tested, {n_sig} significant (q<0.05)")
                    all_results.append(df)
                else:
                    print("FAILED")

    if not all_results:
        print("No results!")
        return

    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv(OUTPUT_DIR / "da_all_tools.csv", index=False)

    # Summary: genera significant by tool × outcome
    sig = combined[combined["qvalue"] < 0.05].copy()

    print(f"\n{'='*60}")
    print("SUMMARY: Significant genera (q < 0.05)")
    print(f"{'='*60}")

    for outcome in METABOLIC_OUTCOMES:
        label = outcome.replace("y_", "").replace("_", " ").title()
        print(f"\n  {label}:")
        out_sig = sig[sig["outcome"] == outcome]
        for tool in TOOLS:
            tool_sig = out_sig[out_sig["tool"] == tool]
            n = len(tool_sig)
            genera = ", ".join(tool_sig["feature"].head(5).tolist())
            suffix = f"... (+{n-5} more)" if n > 5 else ""
            print(f"    {tool:10s}: {n:3d} genera  {genera}{suffix}")

    # Consensus: genera significant in 3+ tools per outcome
    print(f"\n{'='*60}")
    print("CONSENSUS: Genera significant in ≥3 tools")
    print(f"{'='*60}")

    consensus_rows = []
    for outcome in METABOLIC_OUTCOMES:
        label = outcome.replace("y_", "").replace("_", " ").title()
        out_sig = sig[sig["outcome"] == outcome]
        genus_tool_count = out_sig.groupby("feature")["tool"].nunique()
        consensus = genus_tool_count[genus_tool_count >= 3].sort_values(ascending=False)

        if len(consensus) > 0:
            print(f"\n  {label}:")
            for genus, n_tools in consensus.items():
                tools_list = out_sig[out_sig["feature"] == genus]["tool"].tolist()
                # Get direction from majority
                genus_data = out_sig[out_sig["feature"] == genus]
                mean_lfc = genus_data["log2fc"].mean()
                direction = "↑" if mean_lfc > 0 else "↓"
                print(f"    {direction} {genus:30s} ({n_tools}/5 tools: {', '.join(tools_list)})")
                consensus_rows.append({
                    "outcome": outcome,
                    "genus": genus,
                    "n_tools": n_tools,
                    "tools": "|".join(tools_list),
                    "mean_log2fc": mean_lfc,
                    "direction": "enriched" if mean_lfc > 0 else "depleted",
                })

    if consensus_rows:
        consensus_df = pd.DataFrame(consensus_rows)
        consensus_df.to_csv(OUTPUT_DIR / "da_consensus_genera.csv", index=False)
        print(f"\nConsensus results saved to {OUTPUT_DIR / 'da_consensus_genera.csv'}")

    print(f"All results saved to {OUTPUT_DIR / 'da_all_tools.csv'}")


if __name__ == "__main__":
    main()

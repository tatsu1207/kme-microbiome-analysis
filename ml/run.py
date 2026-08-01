"""
Main runner: data join → feature engineering → model comparison.

Usage:
    cd /path/to/kme-microbiome-analysis
    python -m ml.run                   # all three sources
    python -m ml.run --source dada2    # DADA2 genus-level
    python -m ml.run --source mothur   # mothur OTU-based
    python -m ml.run --source asv      # DADA2 ASV-level (no genus collapse)
"""
import argparse

import numpy as np
import pandas as pd

from ml.config import OUTPUT_DIR, TOP_N_ASVS
from ml.data_join import (
    build_joined_dataset,
    clr_transform,
    load_asv_abundance,
    load_kmedata,
    compute_outcomes,
    filter_questionnaire_cols,
    select_top_features,
)
from ml.data_join_mothur import build_joined_dataset_mothur
from ml.features import (
    assign_blocks,
    build_feature_matrices,
    print_feature_summary,
    process_questionnaire,
)
from ml.model import make_pipeline, run_all
from ml.config import MIN_AVAILABILITY


def build_joined_dataset_asv() -> dict[str, pd.DataFrame]:
    """Build joined dataset using raw ASVs instead of genus-collapsed."""
    print("Loading KMEDATA questionnaire...")
    kme = load_kmedata()

    print("Loading DADA2 ASV abundances (no genus collapse)...")
    asv_rel = load_asv_abundance()
    print(f"ASVs after prevalence filter: {asv_rel.shape[1]}")

    paired_ids = kme.index.intersection(asv_rel.index)
    print(f"Paired subjects: {len(paired_ids)}")

    kme = kme.loc[paired_ids]
    asv_rel = asv_rel.loc[paired_ids]

    ds_mask = kme.index.str.contains("_DS_")
    kme = kme.loc[ds_mask]
    asv_rel = asv_rel.loc[ds_mask]
    paired_ids = kme.index
    print(f"DS paired subjects: {len(paired_ids)}")

    # Select top N ASVs by variance, CLR-transform
    print(f"Selecting top {TOP_N_ASVS} ASVs by variance...")
    asv_top = select_top_features(asv_rel, TOP_N_ASVS)
    asv_clr = clr_transform(asv_top)
    asv_clr.columns = ["asv__" + c for c in asv_clr.columns]

    # Questionnaire
    print("Filtering questionnaire columns...")
    good_cols = filter_questionnaire_cols(kme, MIN_AVAILABILITY)
    print(f"Questionnaire columns with >={MIN_AVAILABILITY*100:.0f}% data: {len(good_cols)}")
    quest = kme[good_cols].copy()

    outcomes = compute_outcomes(kme)
    meta = kme[["YEAR", "SEX", "AGE"]].copy()
    meta["AGE"] = pd.to_numeric(meta["AGE"], errors="coerce")

    result = {
        "questionnaire": quest,
        "genus_clr": asv_clr,  # reuse same key so build_feature_matrices works
        "outcomes": outcomes,
        "meta": meta,
    }

    print(f"\n=== Dataset Summary (ASV-level) ===")
    print(f"Subjects:       {len(paired_ids)}")
    print(f"ASV features:   {asv_clr.shape[1]}")
    print(f"Quest cols:     {quest.shape[1]}")

    return result


def run_pipeline(source: str) -> tuple:
    """Run the full pipeline for a given data source. Returns all_results."""
    print("=" * 60)
    print(f"STEP 1: Data joining ({source})")
    print("=" * 60)

    if source == "dada2":
        data = build_joined_dataset()
    elif source == "mothur":
        data = build_joined_dataset_mothur()
    elif source == "asv":
        data = build_joined_dataset_asv()

    # Step 2: Feature engineering
    print("\n" + "=" * 60)
    print(f"STEP 2: Feature engineering ({source})")
    print("=" * 60)
    quest_processed = process_questionnaire(data["questionnaire"], data["meta"])
    block_assignments = assign_blocks(quest_processed)

    genus_clr = data["genus_clr"]

    if "pathways" in data:
        pathways = data["pathways"]
        feature_matrices = build_feature_matrices(quest_processed, genus_clr, pathways)
    else:
        empty_pw = pd.DataFrame(index=genus_clr.index)
        feature_matrices = build_feature_matrices(quest_processed, genus_clr, empty_pw)

    print_feature_summary(feature_matrices, block_assignments)

    # Step 3: Model comparison
    print("\n" + "=" * 60)
    print(f"STEP 3: Model comparison ({source})")
    print("=" * 60)
    all_results = run_all(feature_matrices, data["outcomes"])

    # Step 4: Feature importance for key outcomes
    suffix = f"_{source}"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    m3 = feature_matrices["M3"]
    for target_name in ["y_obesity", "y_digestive_symptoms", "y_abnormal_stool", "y_poor_gut"]:
        if target_name not in data["outcomes"].columns:
            continue
        y_t = data["outcomes"][target_name]
        valid = y_t.notna()
        if valid.sum() < 50:
            continue
        print(f"\n{'='*60}")
        print(f"Feature importance (M3, {target_name}, {source})")
        print(f"{'='*60}")
        X_fit = m3.loc[valid]
        y_fit = y_t.loc[valid].astype(int)
        pipe = make_pipeline()
        pipe.fit(X_fit, y_fit)
        importance = pd.Series(
            pipe.named_steps["clf"].feature_importances_, index=X_fit.columns
        ).sort_values(ascending=False)
        print(f"\nTop 20 features by XGBoost gain importance:")
        for i, (feat, val) in enumerate(importance.head(20).items()):
            print(f"  {i+1:2d}. {feat:<50s} {val:.4f}")
        importance.to_csv(OUTPUT_DIR / f"feature_importance_{target_name}_m3{suffix}.csv")

    # Save summary
    summary_rows = []
    for target, r in all_results.items():
        summary_rows.append({
            "outcome": target,
            "AUC_M1": r.get("M1", {}).get("auc"),
            "AUC_M2": r.get("M2", {}).get("auc"),
            "AUC_M3": r.get("M3", {}).get("auc"),
            "delta_AUC_M3_vs_M1": r.get("M3_vs_M1", {}).get("delta_auc"),
            "p_M3_vs_M1": r.get("M3_vs_M1", {}).get("p"),
            "delta_AUC_M3_vs_M2": r.get("M3_vs_M2", {}).get("delta_auc"),
            "p_M3_vs_M2": r.get("M3_vs_M2", {}).get("p"),
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTPUT_DIR / f"model_comparison_summary{suffix}.csv", index=False)
    print(f"\nSummary saved to model_comparison_summary{suffix}.csv")

    return all_results, summary_df


def print_comparison_3way(summaries: dict[str, pd.DataFrame]):
    """Print side-by-side comparison of all sources."""
    print("\n" + "=" * 90)
    print("COMPARISON: Genus (DADA2) vs ASV (DADA2) vs OTU (Mothur)")
    print("=" * 90)
    print(f"{'Outcome':<25} {'-- Genus --':>14} {'--- ASV ---':>14} {'-- Mothur --':>14}")
    print(f"{'':25} {'M1':>6} {'M2':>6}  {'M1':>6} {'M2':>6}  {'M1':>6} {'M2':>6}")
    print("-" * 90)

    # Merge all summaries
    merged = summaries["dada2"][["outcome", "AUC_M1", "AUC_M2", "AUC_M3"]].copy()
    merged.columns = ["outcome", "M1_genus", "M2_genus", "M3_genus"]

    for src in ["asv", "mothur"]:
        if src in summaries:
            s = summaries[src][["outcome", "AUC_M1", "AUC_M2", "AUC_M3"]].copy()
            s.columns = ["outcome", f"M1_{src}", f"M2_{src}", f"M3_{src}"]
            merged = merged.merge(s, on="outcome", how="outer")

    for _, row in merged.iterrows():
        line = f"{row['outcome']:<25}"
        for src in ["genus", "asv", "mothur"]:
            m1 = row.get(f"M1_{src}", np.nan)
            m2 = row.get(f"M2_{src}", np.nan)
            m1s = f"{m1:.3f}" if not np.isnan(m1) else "  n/a"
            m2s = f"{m2:.3f}" if not np.isnan(m2) else "  n/a"
            line += f"  {m1s} {m2s}"
        print(line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        choices=["dada2", "mothur", "asv", "both", "all"],
        default="all",
    )
    args = parser.parse_args()

    if args.source in ("dada2", "mothur", "asv"):
        run_pipeline(args.source)
    elif args.source == "both":
        _, s1 = run_pipeline("dada2")
        _, s2 = run_pipeline("mothur")
        print_comparison_3way({"dada2": s1, "mothur": s2})
    else:
        # Run all three
        summaries = {}
        for src in ["dada2", "asv", "mothur"]:
            print("\n\n" + "#" * 80)
            print(f"# RUNNING {src.upper()} PIPELINE")
            print("#" * 80 + "\n")
            _, summaries[src] = run_pipeline(src)
        print_comparison_3way(summaries)


if __name__ == "__main__":
    main()

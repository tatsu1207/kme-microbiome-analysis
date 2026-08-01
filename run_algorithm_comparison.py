"""
Run logistic regression, random forest, and XGBoost on all outcomes.
Shows that the M3≈M1 finding is algorithm-agnostic.

Usage:
    cd /home/unnot/github/health-check/microbiome_report
    python -m ml.run_algorithm_comparison
"""
import numpy as np
import pandas as pd
from pathlib import Path

from ml.config import OUTPUT_DIR, RANDOM_STATE
from ml.data_join import build_joined_dataset
from ml.features import build_feature_matrices, process_questionnaire, assign_blocks
from ml.model import run_all, CLASSIFIERS

ALGORITHMS = ["xgboost", "logistic_regression", "random_forest"]
ALGO_LABELS = {
    "xgboost": "XGBoost",
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
}


def main():
    print("Building feature matrices...")
    data = build_joined_dataset()
    quest_processed = process_questionnaire(data["questionnaire"], data["meta"])
    genus_clr = data["genus_clr"]
    pathways = data.get("pathways", pd.DataFrame(index=genus_clr.index))
    feature_matrices = build_feature_matrices(quest_processed, genus_clr, pathways)
    outcomes = data["outcomes"]

    all_algo_results = {}

    for algo in ALGORITHMS:
        print(f"\n{'#'*70}")
        print(f"# ALGORITHM: {ALGO_LABELS[algo]}")
        print(f"{'#'*70}")

        results = run_all(feature_matrices, outcomes, algorithm=algo)
        all_algo_results[algo] = results

    # Build comparison table
    rows = []
    targets = [
        "y_obesity", "y_hypertension", "y_prediabetes", "y_liver", "y_multimorbidity",
        "y_digestive_symptoms", "y_multi_digestive", "y_abnormal_stool",
        "y_bowel_difficulty", "y_poor_gut",
    ]

    for target in targets:
        label = target.replace("y_", "").replace("_", " ").title()
        row = {"Outcome": label}

        for algo in ALGORITHMS:
            alabel = ALGO_LABELS[algo]
            r = all_algo_results[algo].get(target, {})
            m1 = r.get("M1", {}).get("auc", np.nan)
            m2 = r.get("M2", {}).get("auc", np.nan)
            m3 = r.get("M3", {}).get("auc", np.nan)
            delta = r.get("M3_vs_M1", {}).get("delta_auc", np.nan)
            p = r.get("M3_vs_M1", {}).get("p", np.nan)

            row[f"{alabel} M1"] = m1
            row[f"{alabel} M2"] = m2
            row[f"{alabel} M3"] = m3
            row[f"{alabel} ΔAUC"] = delta
            row[f"{alabel} p"] = p

        rows.append(row)

    df = pd.DataFrame(rows)
    out_path = OUTPUT_DIR / "algorithm_comparison.csv"
    df.to_csv(out_path, index=False)

    # Print summary
    print(f"\n{'='*90}")
    print("ALGORITHM COMPARISON SUMMARY")
    print(f"{'='*90}")
    print(f"{'Outcome':<20}", end="")
    for algo in ALGORITHMS:
        alabel = ALGO_LABELS[algo][:8]
        print(f"  {alabel:>8} M1  {alabel:>8} M3  {'ΔAUC':>6}  {'p':>6}", end="")
    print()
    print("-" * 110)

    for _, row in df.iterrows():
        print(f"{row['Outcome']:<20}", end="")
        for algo in ALGORITHMS:
            alabel = ALGO_LABELS[algo]
            m1 = row[f"{alabel} M1"]
            m3 = row[f"{alabel} M3"]
            delta = row[f"{alabel} ΔAUC"]
            p = row[f"{alabel} p"]
            sig = "*" if p < 0.05 else " "
            print(f"  {m1:>11.3f}  {m3:>11.3f}  {delta:>+6.3f}  {p:>5.3f}{sig}", end="")
        print()

    # Check: any algorithm shows M3 > M1 significantly?
    any_sig = False
    for algo in ALGORITHMS:
        alabel = ALGO_LABELS[algo]
        for _, row in df.iterrows():
            if row[f"{alabel} p"] < 0.05:
                any_sig = True
                print(f"\n  !! SIGNIFICANT: {row['Outcome']} with {alabel}: "
                      f"ΔAUC={row[f'{alabel} ΔAUC']:+.4f}, p={row[f'{alabel} p']:.4f}")

    if not any_sig:
        print(f"\n  No algorithm shows significant M3 vs M1 improvement (all p > 0.05)")
        print(f"  The finding is ALGORITHM-AGNOSTIC.")

    print(f"\nSaved to {out_path}")

    # Also save as a cleaner supplementary table
    sup_rows = []
    for _, row in df.iterrows():
        for algo in ALGORITHMS:
            alabel = ALGO_LABELS[algo]
            sup_rows.append({
                "Outcome": row["Outcome"],
                "Algorithm": alabel,
                "AUC M1 (Quest.)": f"{row[f'{alabel} M1']:.3f}",
                "AUC M2 (Microbiome)": f"{row[f'{alabel} M2']:.3f}",
                "AUC M3 (Combined)": f"{row[f'{alabel} M3']:.3f}",
                "ΔAUC (M3-M1)": f"{row[f'{alabel} ΔAUC']:+.4f}",
                "p (DeLong)": f"{row[f'{alabel} p']:.4f}",
            })

    sup_df = pd.DataFrame(sup_rows)
    sup_path = Path("vault/health/results/tableS10_algorithm_comparison.csv")
    sup_df.to_csv(sup_path, index=False)
    print(f"Supplementary table saved to {sup_path}")


if __name__ == "__main__":
    main()

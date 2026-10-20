"""
Longevity Village Analysis: composite fiber-degrader index.

Index = summed relative abundance of 10 fiber-degrading genera. Compared
between RC and DS adults aged 60+ (Mann-Whitney U; Cohen's d using the
unweighted average of the two group variances), in the full cohort and in
the 60-69 age-matched subset, and as a 7-genus index without the three
DS-60+ outlier genera (Roseburia, Coprococcus, [Eubacterium] eligens group).

Within the RC, Spearman correlations of summed lactobacilli
(Limosilactobacillus + Lactobacillus + Ligilactobacillus) with Megasphaera
and with the 10-genus index are reported for two definitions of the
lactobacilli sum: summed relative abundance and summed CLR values.

Usage (set paths in ml/longevity_config.py first):
    cd /path/to/kme-microbiome-analysis
    python -m ml.longevity_fiber_index
"""
import json

import numpy as np
import pandas as pd
from scipy import stats

from ml.data_join import clr_transform
from ml.longevity_config import LONGEVITY_OUT
from ml.longevity_muan import load_all_data

OUT = LONGEVITY_OUT

FIBER_GENERA = [
    "Lachnospira",                    # pectin degrader
    "Monoglobus",                     # pectin specialist
    "Lachnoclostridium",              # Lachnospiraceae fiber fermenter
    "Lachnospiraceae UCG-004",        # Lachnospiraceae fiber fermenter
    "[Eubacterium] siraeum group",    # resistant starch degrader
    "Oscillospira",                   # fiber-rich diet associated
    "Prevotella",                     # plant polysaccharide degrader
    "Roseburia",                      # butyrate producer from dietary fiber
    "Coprococcus",                    # butyrate producer from dietary fiber
    "[Eubacterium] eligens group",    # pectin degrader
]
DS60_OUTLIER_GENERA = ["Roseburia", "Coprococcus", "[Eubacterium] eligens group"]
LACTOBACILLI = ["Limosilactobacillus", "Lactobacillus", "Ligilactobacillus"]


def cohens_d(a, b):
    """(mean_a - mean_b) / sqrt((var_a + var_b) / 2), sample variances."""
    return (np.mean(a) - np.mean(b)) / np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)


def compare(index, rc_idx, ds_idx):
    r, d = index.loc[rc_idx].values, index.loc[ds_idx].values
    _, p = stats.mannwhitneyu(r, d, alternative="two-sided")
    return {"rc_n": len(r), "ds_n": len(d), "rc_mean": float(r.mean()), "ds_mean": float(d.mean()),
            "mw_P": float(p), "cohens_d": float(cohens_d(r, d))}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    meta, genus_rel, _ = load_all_data()
    genus_clr = clr_transform(genus_rel)

    rc = meta.index[meta["cohort"] == "Muan"]
    ds60 = meta.index[(meta["cohort"] == "DS") & (meta["age"] >= 60)]
    rc_6069 = meta.index[(meta["cohort"] == "Muan") & meta["age"].between(60, 69)]
    ds_6069 = meta.index[(meta["cohort"] == "DS") & meta["age"].between(60, 69)]

    genera7 = [g for g in FIBER_GENERA if g not in DS60_OUTLIER_GENERA]
    index10 = genus_rel[FIBER_GENERA].sum(axis=1)
    index7 = genus_rel[genera7].sum(axis=1)

    results = {
        "index10_rc_vs_ds60": compare(index10, rc, ds60),
        "index10_age_60_69": compare(index10, rc_6069, ds_6069),
        "index7_rc_vs_ds60": compare(index7, rc, ds60),
        "index7_age_60_69": compare(index7, rc_6069, ds_6069),
    }

    # Within-RC correlations, for both definitions of summed lactobacilli
    # (Megasphaera on the same scale as the lactobacilli sum)
    lacto = {"relative_abundance": (genus_rel.loc[rc, LACTOBACILLI].sum(axis=1), genus_rel.loc[rc, "Megasphaera"]),
             "clr": (genus_clr.loc[rc, LACTOBACILLI].sum(axis=1), genus_clr.loc[rc, "Megasphaera"])}
    within = {}
    for name, (lac, megasphaera) in lacto.items():
        for target, y in [("Megasphaera", megasphaera), ("index10", index10.loc[rc])]:
            rho, p = stats.spearmanr(lac, y)
            within[f"lactobacilli_{name}_vs_{target}"] = {"rho": float(rho), "P": float(p)}
    results["within_rc_spearman"] = within

    pd.DataFrame({"cohort": meta["cohort"], "age": meta["age"],
                  "fiber_index10": index10, "fiber_index7": index7}).to_csv(
        OUT / "fiber_degrader_index.csv", index_label="sample_id")
    with open(OUT / "fiber_degrader_index_summary.json", "w") as f:
        json.dump(results, f, indent=2)

    for name, r in results.items():
        if name == "within_rc_spearman":
            for k, v in r.items():
                print(f"  {k}: rho = {v['rho']:+.3f}, P = {v['P']:.2e}")
        else:
            print(f"  {name}: RC n={r['rc_n']}, DS n={r['ds_n']}, d = {r['cohens_d']:+.2f}, P = {r['mw_P']:.1e}")
    print(f"  Saved to {OUT}")


if __name__ == "__main__":
    main()

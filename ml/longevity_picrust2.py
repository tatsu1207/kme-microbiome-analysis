"""
Longevity Village Analysis: PICRUSt2 functional comparisons (Fig. S5).

Age-restricted comparison of DS 60-69 vs RC 60-69 (plus DS young, 20-39, for
the three-group classification) on PICRUSt2 KO and MetaCyc pathway
predictions, from the unrarefied run and the run on the ASV table rarefied to
10,000 reads:

  - weighted NSTI (prediction quality)
  - butyrate-synthesis KOs: kinase route (ptb K00634, buk K00929, bcd K00248)
    and CoA-transferase route (but K01034/K01035), each and summed
  - Mann-Whitney U on every KO / pathway, BH-FDR across the features tested
  - fermentation, glycolysis and lactose catabolism pathways
  - three-group pathway classification: "aging" = DS young vs DS 60-69 and
    "cohort" = DS 60-69 vs RC 60-69 at q < 0.05

Features are tested if present (> 0) in at least 10% of the samples of at
least one of the groups being compared. Fold change = RC mean / DS mean.

The rarefied input is produced with --make-rarefied-input (all ASV reads,
10,000 per sample, without replacement); run picrust2_pipeline.py on it with
-o pointing at PICRUST2_RAREFIED_DIR. A new rarefaction draw gives slightly
different predictions from the published run.

Usage (set paths in ml/longevity_config.py first):
    cd /path/to/kme-microbiome-analysis
    python -m ml.longevity_picrust2 --make-rarefied-input   # once, then run PICRUSt2
    python -m ml.longevity_picrust2
"""
import argparse
import json

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from ml.config import RANDOM_STATE
from ml.longevity_config import (
    LONGEVITY_ASV_TABLE, LONGEVITY_OUT, PICRUST2_DIR, PICRUST2_RAREFIED_DIR,
)
from ml.longevity_muan import load_all_data

OUT = LONGEVITY_OUT
RAREFY_DEPTH = 10_000
MIN_GROUP_PREVALENCE = 0.10

KINASE_KOS = {"K00634": "ptb", "K00929": "buk", "K00248": "bcd"}
COA_TRANSFERASE_KOS = {"K01034": "but (A)", "K01035": "but (B)"}
BUTYRATE_KOS = {**KINASE_KOS, **COA_TRANSFERASE_KOS}
KEY_PATHWAYS = ["FERMENTATION-PWY", "GLYCOLYSIS", "LACTOSECAT-PWY"]


# ===================================================================
# INPUT
# ===================================================================

def make_rarefied_input(seed=RANDOM_STATE):
    """Rarefy ASV counts to 10,000 reads; write PICRUSt2 inputs."""
    asv = pd.read_csv(LONGEVITY_ASV_TABLE, sep="\t", index_col=0)
    seqs = asv["sequence"]
    counts = asv.drop(columns=["sequence"])
    keep = counts.columns[counts.sum(axis=0) >= RAREFY_DEPTH]
    rng = np.random.default_rng(seed)
    rarefied = pd.DataFrame(
        {s: rng.multivariate_hypergeometric(counts[s].values.astype(np.int64), RAREFY_DEPTH)
         for s in keep}, index=counts.index)
    rarefied = rarefied.loc[rarefied.sum(axis=1) > 0]

    OUT.mkdir(parents=True, exist_ok=True)
    rarefied.to_csv(OUT / "picrust2_counts_rarefied.tsv", sep="\t", index_label="ASV_ID")
    with open(OUT / "picrust2_seqs_rarefied.fasta", "w") as f:
        for asv_id in rarefied.index:
            f.write(f">{asv_id}\n{seqs[asv_id]}\n")
    print(f"  Rarefied {len(keep)} of {counts.shape[1]} samples to {RAREFY_DEPTH} reads "
          f"({len(rarefied)} ASVs). Next:\n"
          f"    picrust2_pipeline.py -s {OUT / 'picrust2_seqs_rarefied.fasta'} "
          f"-i {OUT / 'picrust2_counts_rarefied.tsv'} -o {PICRUST2_RAREFIED_DIR}")


def load_picrust2(run_dir):
    """KO and pathway tables (samples x features) and weighted NSTI."""
    ko = pd.read_csv(run_dir / "KO_metagenome_out/pred_metagenome_unstrat.tsv.gz",
                     sep="\t", index_col=0).T
    ko.columns = [c.replace("ko:", "") for c in ko.columns]
    pathways = pd.read_csv(run_dir / "pathways_out/path_abun_unstrat.tsv.gz",
                           sep="\t", index_col=0).T
    nsti = pd.read_csv(run_dir / "KO_metagenome_out/weighted_nsti.tsv.gz",
                       sep="\t", index_col=0).iloc[:, 0]
    return ko, pathways, nsti


# ===================================================================
# TESTS
# ===================================================================

def prevalent(table, groups):
    """Features present in >= 10% of samples of at least one group."""
    mask = np.zeros(table.shape[1], dtype=bool)
    for g in groups:
        mask |= ((table.loc[g] > 0).mean() >= MIN_GROUP_PREVALENCE).values
    return table.columns[mask]


def mw_table(table, a, b, features):
    """Mann-Whitney a vs b per feature; fold change = mean(a) / mean(b)."""
    p = np.array([stats.mannwhitneyu(table.loc[a, f], table.loc[b, f],
                                     alternative="two-sided").pvalue for f in features])
    mean_a, mean_b = table.loc[a, features].mean(), table.loc[b, features].mean()
    df = pd.DataFrame({"mean_a": mean_a.values, "mean_b": mean_b.values,
                       "fc": (mean_a / mean_b).values, "p": p}, index=features)
    df["q"] = multipletests(df["p"], method="fdr_bh")[1]
    return df.sort_values("p")


def summed(table, a, b, kos):
    x, y = table.loc[a, kos].sum(axis=1), table.loc[b, kos].sum(axis=1)
    return {"fc": float(x.mean() / y.mean()), "pct_lower": float(100 * (1 - x.mean() / y.mean())),
            "P": float(stats.mannwhitneyu(x, y, alternative="two-sided").pvalue)}


def analyse_run(label, run_dir, groups):
    ko, pathways, nsti = load_picrust2(run_dir)
    young, ds, rc = ([s for s in groups[k] if s in ko.index] for k in ("ds_young", "ds_6069", "rc_6069"))

    ko_da = mw_table(ko, rc, ds, prevalent(ko, [ds, rc]))
    pw_da = mw_table(pathways, rc, ds, prevalent(pathways, [ds, rc]))
    ko_da.to_csv(OUT / f"picrust2_{label}_KO_rc_vs_ds_6069.csv", index_label="KO")
    pw_da.to_csv(OUT / f"picrust2_{label}_pathway_rc_vs_ds_6069.csv", index_label="pathway")

    # Three-group pathway classification
    feats = prevalent(pathways, [young, ds, rc])
    aging = mw_table(pathways, young, ds, feats)["q"] < 0.05
    cohort = mw_table(pathways, ds, rc, feats)["q"] < 0.05
    cls = pd.DataFrame({"aging": aging, "cohort": cohort.reindex(aging.index)})
    cls.to_csv(OUT / f"picrust2_{label}_pathway_3group.csv", index_label="pathway")

    nsti_rc, nsti_ds = nsti.reindex(rc).dropna(), nsti.reindex(ds).dropna()
    return {
        "n": {"ds_young": len(young), "ds_6069": len(ds), "rc_6069": len(rc)},
        "nsti": {"ds_mean": float(nsti_ds.mean()), "rc_mean": float(nsti_rc.mean()),
                 "P": float(stats.mannwhitneyu(nsti_rc, nsti_ds, alternative="two-sided").pvalue)},
        "n_ko_tested": len(ko_da), "n_pathways_tested": len(pw_da),
        "butyrate_kos": {f"{k} {v}": ko_da.loc[k, ["fc", "p", "q"]].to_dict()
                         for k, v in BUTYRATE_KOS.items() if k in ko_da.index},
        "butyrate_total_5_kos": summed(ko, rc, ds, list(BUTYRATE_KOS)),
        "key_pathways": {p: pw_da.loc[p, ["fc", "p", "q"]].to_dict() for p in KEY_PATHWAYS if p in pw_da.index},
        "three_group": {
            "n_pathways": len(cls),
            "cohort_only": int((cls.cohort & ~cls.aging).sum()),
            "both": int((cls.cohort & cls.aging).sum()),
            "aging_only": int((cls.aging & ~cls.cohort).sum()),
            "neither": int((~cls.aging & ~cls.cohort).sum()),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--make-rarefied-input", action="store_true",
                        help="write the rarefied PICRUSt2 input and exit")
    args = parser.parse_args()
    if args.make_rarefied_input:
        make_rarefied_input()
        return

    OUT.mkdir(parents=True, exist_ok=True)
    meta, _, _ = load_all_data()
    groups = {
        "ds_young": meta.index[(meta["cohort"] == "DS") & (meta["age"] < 40)],
        "ds_6069": meta.index[(meta["cohort"] == "DS") & meta["age"].between(60, 69)],
        "rc_6069": meta.index[(meta["cohort"] == "Muan") & meta["age"].between(60, 69)],
    }

    summary = {}
    for label, run_dir in [("unrarefied", PICRUST2_DIR), ("rarefied", PICRUST2_RAREFIED_DIR)]:
        if not (run_dir / "pathways_out").exists():
            print(f"  {label}: {run_dir} not found, skipping")
            continue
        summary[label] = s = analyse_run(label, run_dir, groups)
        t = s["three_group"]
        print(f"  {label}: n = {s['n']}; {s['n_ko_tested']} KOs, {s['n_pathways_tested']} pathways tested; "
              f"NSTI P = {s['nsti']['P']:.2f}")
        print(f"    butyrate (5 KOs) {s['butyrate_total_5_kos']['pct_lower']:.0f}% lower in RC, "
              f"P = {s['butyrate_total_5_kos']['P']:.1e}")
        for k, v in s["butyrate_kos"].items():
            print(f"    {k}: FC = {v['fc']:.2f}, P = {v['p']:.3f}, q = {v['q']:.3f}")
        for k, v in s["key_pathways"].items():
            print(f"    {k}: FC = {v['fc']:.2f}, q = {v['q']:.3f}")
        print(f"    3-group: cohort-only {t['cohort_only']} of {t['n_pathways']}, both {t['both']}, "
              f"aging-only {t['aging_only']}, neither {t['neither']}")

    with open(OUT / "picrust2_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved to {OUT}")


if __name__ == "__main__":
    main()

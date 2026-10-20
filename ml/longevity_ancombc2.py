"""
Longevity Village Analysis: ANCOM-BC2 sensitivity analysis (Table S3).

Runs ANCOM-BC2 (r_scripts/run_ancombc2_longevity.R) on genus-level counts for
RC vs DS adults aged 60+, with sex as a covariate, then compares the result
with the primary Mann-Whitney analysis within the fixed 153-genus set.

ANCOM-BC2 applies its own 10% prevalence filter within the analysed samples,
so its output can include a few genera outside the 153-genus set; q-values are
BH-adjusted across all genera it tests. The comparison counts use only the
genera in the 153-genus set.

Requires R with ANCOMBC, phyloseq, optparse and jsonlite.

Usage (set paths in ml/longevity_config.py first):
    cd /path/to/kme-microbiome-analysis
    python -m ml.longevity_ancombc2 [--threads 8]
"""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from ml.config import RANDOM_STATE
from ml.longevity_config import LONGEVITY_OUT
from ml.longevity_muan import get_age_groups, load_all_data
from ml.longevity_sensitivity import load_genus_counts
from ml.longevity_table_s1 import table_s1

OUT = LONGEVITY_OUT
R_SCRIPT = Path(__file__).resolve().parent / "r_scripts" / "run_ancombc2_longevity.R"


def write_inputs(meta, groups, counts_path, meta_path):
    """Genus counts and metadata for RC + DS 60+ samples with known sex."""
    idx = groups.index[groups.isin(["RC elderly", "DS elderly"]) & meta["sex"].isin(["F", "M"])]
    counts = load_genus_counts().loc[idx]
    counts.T.to_csv(counts_path, sep="\t")
    pd.DataFrame({
        "SampleID": idx,
        "cohort": np.where(groups.loc[idx] == "RC elderly", "RC", "DS"),
        "sex": meta.loc[idx, "sex"].values,
    }).to_csv(meta_path, sep="\t", index=False)
    return idx


def compare_with_mw(ancom, primary):
    """Agreement between ANCOM-BC2 and Mann-Whitney within the 153-genus set."""
    tested = ancom[ancom["genus"].isin(primary["genus"])]
    mw_sig = primary[primary["q_value"] < 0.05]
    m = mw_sig.merge(tested, on="genus", how="left")
    testable = m[m["lfc"].notna()]
    confirmed = testable[testable["diff"].astype(bool)]
    same_sign = np.sign(confirmed["clr_diff"]) == np.sign(confirmed["lfc"])
    ancom_sig = set(tested.loc[tested["diff"].astype(bool), "genus"])
    return {
        "ancom_tested_total": len(ancom),
        "ancom_tested_in_153": len(tested),
        "excluded_by_filtering": sorted(set(primary["genus"]) - set(ancom["genus"])),
        "ancom_significant_in_153": len(ancom_sig),
        "primary_significant": len(mw_sig),
        "primary_testable": len(testable),
        "confirmed": len(confirmed),
        "confirmed_passed_ss": int(confirmed["passed_ss"].astype(bool).sum()),
        "mw_only": len(mw_sig) - len(confirmed),
        "ancom_only": len(ancom_sig - set(mw_sig["genus"])),
        "same_sign_confirmed": int(same_sign.sum()),
        "opposite_sign": sorted(confirmed.loc[~same_sign, "genus"]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    meta, genus_rel, _ = load_all_data()
    groups = get_age_groups(meta)

    counts_path, meta_path = OUT / "ancombc2_counts.tsv", OUT / "ancombc2_metadata.tsv"
    result_path = OUT / "tableS3_ancombc2.tsv"
    idx = write_inputs(meta, groups, counts_path, meta_path)
    print(f"  ANCOM-BC2 input: {len(idx)} samples")

    subprocess.run(["Rscript", str(R_SCRIPT), "--counts", str(counts_path),
                    "--metadata", str(meta_path), "--output", str(result_path),
                    "--threads", str(args.threads), "--seed", str(RANDOM_STATE)], check=True)

    ancom = pd.read_csv(result_path, sep="\t")
    ancom.to_csv(OUT / "tableS3_ancombc2.csv", index=False)
    summary = compare_with_mw(ancom, table_s1(meta, genus_rel))
    with open(OUT / "ancombc2_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"  Saved to {OUT}")


if __name__ == "__main__":
    main()

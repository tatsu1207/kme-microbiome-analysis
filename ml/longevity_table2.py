"""
Longevity Village Analysis: Table 2 (PERMANOVA and PERMDISP).

Bray-Curtis dissimilarities on genus relative abundance, 999 permutations.
PERMANOVA (pseudo-F, R², P) for cohort, sex and the four age groups, and for
two restricted cohort comparisons (RC vs DS 60+; both cohorts aged 60-69).
PERMDISP (distance to the group spatial median, as in vegan::betadisper)
for the three cohort comparisons.

Requires scikit-bio for PERMDISP.

Usage (set paths in ml/longevity_config.py first):
    cd /path/to/kme-microbiome-analysis
    python -m ml.longevity_table2
"""
import numpy as np
import pandas as pd
from skbio import DistanceMatrix
from skbio.stats.distance import permdisp

from ml.config import RANDOM_STATE
from ml.longevity_config import LONGEVITY_OUT
from ml.longevity_muan import _bray_curtis, _permanova, get_age_groups, load_all_data

OUT = LONGEVITY_OUT
N_PERM = 999


def permanova_r2(f_stat, n, k):
    """R² = SS_between / SS_total, recovered from the pseudo-F."""
    return f_stat * (k - 1) / (f_stat * (k - 1) + (n - k))


def run_factor(name, genus_rel, labels, with_permdisp):
    labels = labels.dropna()
    dist = _bray_curtis(genus_rel.loc[labels.index])
    k, n = labels.nunique(), len(labels)

    np.random.seed(RANDOM_STATE)
    f_stat, p = _permanova(dist, labels.values, n_perm=N_PERM)
    row = {"factor": name, "permanova_F": f_stat, "R2_pct": 100 * permanova_r2(f_stat, n, k),
           "permanova_P": p, "permdisp_F": np.nan, "permdisp_P": np.nan, "n": n}

    if with_permdisp:
        res = permdisp(DistanceMatrix(dist, ids=list(labels.index)), labels.values,
                       test="median", permutations=N_PERM, seed=RANDOM_STATE)
        row.update({"permdisp_F": res["test statistic"], "permdisp_P": res["p-value"]})
    return row


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    meta, genus_rel, _ = load_all_data()
    groups = get_age_groups(meta)

    cohort = meta["cohort"].map({"DS": "DS", "Muan": "RC"})
    older = groups.isin(["DS elderly", "RC elderly"])
    age_6069 = meta["age"].between(60, 69)

    rows = [
        run_factor("Cohort (DS vs RC)", genus_rel, cohort, True),
        run_factor("Sex", genus_rel, meta["sex"].where(meta["sex"].isin(["F", "M"])), False),
        run_factor("Age group (4 levels)", genus_rel, groups[groups != ""], False),
        run_factor("Cohort (RC vs DS older adults)", genus_rel, cohort[older], True),
        run_factor("Cohort (RC vs DS, ages 60-69)", genus_rel, cohort[age_6069], True),
    ]
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "table2_permanova_permdisp.csv", index=False)

    for r in rows:
        disp = (f"PERMDISP F = {r['permdisp_F']:.1f}, P = {r['permdisp_P']:.3f}"
                if not np.isnan(r["permdisp_F"]) else "PERMDISP not computed")
        print(f"  {r['factor']}: n = {r['n']}, PERMANOVA F = {r['permanova_F']:.2f}, "
              f"R² = {r['R2_pct']:.2f}%, P = {r['permanova_P']:.3f}; {disp}")
    print(f"  Saved to {OUT}")


if __name__ == "__main__":
    main()

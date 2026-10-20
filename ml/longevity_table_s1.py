"""
Longevity Village Analysis: Table S1 (differential abundance, RC vs DS 60+).

153 genera at >=10% prevalence across all 2,084 samples; CLR (pseudocount 1e-6),
two-sided Mann-Whitney U with BH-FDR, Cliff's delta, and prevalence per group.
Same statistics as fig4_differential_abundance in longevity_muan.py, plus
prevalence columns and the formatted supplementary table.

Usage (set paths in ml/longevity_config.py first):
    cd /path/to/kme-microbiome-analysis
    python -m ml.longevity_table_s1
"""
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from ml.data_join import clr_transform
from ml.longevity_config import LONGEVITY_OUT
from ml.longevity_muan import get_age_groups, load_all_data

OUT = LONGEVITY_OUT


def table_s1(meta, genus_rel):
    """Full-precision differential abundance results, sorted by P."""
    groups = get_age_groups(meta)
    rc = groups[groups == "RC elderly"].index
    ds = groups[groups == "DS elderly"].index

    genus_clr = clr_transform(genus_rel)
    prevalence = (genus_rel > 0).mean(axis=0)
    genera = prevalence[prevalence >= 0.10].index.tolist()

    rows = []
    for genus in genera:
        r, d = genus_clr.loc[rc, genus].values, genus_clr.loc[ds, genus].values
        _, p = stats.mannwhitneyu(r, d, alternative="two-sided")
        delta = (np.sum(r[:, None] > d[None, :])
                 - np.sum(r[:, None] < d[None, :])) / (len(r) * len(d))
        rows.append({
            "genus": genus,
            "clr_diff": r.mean() - d.mean(),
            "mean_rc": r.mean(),
            "mean_ds60": d.mean(),
            "p_value": p,
            "cliffs_delta": delta,
            "prev_ds60": 100 * (genus_rel.loc[ds, genus] > 0).mean(),
            "prev_rc": 100 * (genus_rel.loc[rc, genus] > 0).mean(),
        })
    df = pd.DataFrame(rows)
    _, df["q_value"], _, _ = multipletests(df["p_value"], method="fdr_bh")
    return df.sort_values("p_value").reset_index(drop=True)


def format_table_s1(df):
    """Rounded, labelled version as published (CLR/delta 2 dp, P/q 4 sig. figs)."""
    sig4 = lambda x: float(f"{x:.4g}")
    return pd.DataFrame({
        "Genus": df["genus"],
        "CLR difference (RC − DS older adults)": df["clr_diff"].round(2),
        "Mean CLR (RC)": df["mean_rc"].round(2),
        "Mean CLR (DS older adults)": df["mean_ds60"].round(2),
        "P value": df["p_value"].map(sig4),
        "FDR q value": df["q_value"].map(sig4),
        "Cliff's delta (RC vs DS older adults)": df["cliffs_delta"].round(2),
        "Prevalence DS older adults (%)": df["prev_ds60"].round(1),
        "Prevalence RC (%)": df["prev_rc"].round(1),
    })


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    meta, genus_rel, _ = load_all_data()
    df = table_s1(meta, genus_rel)
    df.to_csv(OUT / "tableS1_full_precision.csv", index=False)
    format_table_s1(df).to_csv(OUT / "tableS1_differential_abundance.csv", index=False)

    sig = df[df["q_value"] < 0.05]
    d = sig["cliffs_delta"].abs()
    print(f"  {len(df)} genera tested, {len(sig)} significant (FDR < 0.05)")
    print(f"  |Cliff's delta| among significant (unrounded): median {d.median():.3f}; "
          f"negligible {(d < 0.147).sum()}, small {((d >= 0.147) & (d < 0.33)).sum()}, "
          f"medium {((d >= 0.33) & (d < 0.474)).sum()}, large {(d >= 0.474).sum()}")
    print(f"  Saved to {OUT}")


if __name__ == "__main__":
    main()

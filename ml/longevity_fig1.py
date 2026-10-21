"""
Longevity Village Analysis: Figure 1 (age distribution by cohort).

Density histograms of age for the DS and RC cohorts (30 bins per cohort).
The sex ratio (66.4% vs 67.3% female; chi-square P printed below) is
reported in Table 1 and the text rather than plotted.

Usage (set paths in ml/longevity_config.py first):
    cd /path/to/kme-microbiome-analysis
    python -m ml.longevity_fig1
"""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from ml.longevity_config import LONGEVITY_OUT
from ml.longevity_muan import load_all_data

OUT = LONGEVITY_OUT

COHORT_COLORS = {"DS": "#4878CF", "RC": "#D65F5F"}


def fig1(meta):
    cohort = meta["cohort"].map({"DS": "DS", "Muan": "RC"})
    fig, ax = plt.subplots(figsize=(7, 5))
    for name in ("DS", "RC"):
        ages = meta.loc[cohort == name, "age"]
        ax.hist(ages, bins=30, density=True, alpha=0.6, color=COHORT_COLORS[name],
                label=f"{name} (n={len(ages):,})")
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Density")
    ax.legend()
    plt.tight_layout()
    fig.savefig(OUT / "fig1_age_distribution.png", dpi=300)
    fig.savefig(OUT / "fig1_age_distribution.pdf")
    plt.close(fig)


def sex_test(meta):
    """Female share per cohort and chi-square test (participants with recorded sex)."""
    known = meta[meta["sex"].isin(["F", "M"])]
    ct = pd.crosstab(known["cohort"], known["sex"])
    _, p, _, _ = stats.chi2_contingency(ct)
    for c in ("DS", "Muan"):
        print(f"  {'RC' if c == 'Muan' else c}: {100 * (meta.loc[meta['cohort'] == c, 'sex'] == 'F').mean():.1f}% female")
    print(f"  Sex distribution, chi-square P = {p:.2f}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    meta, _, _ = load_all_data()
    fig1(meta)
    sex_test(meta)
    print(f"  Saved fig1_age_distribution.pdf/.png to {OUT}")


if __name__ == "__main__":
    main()

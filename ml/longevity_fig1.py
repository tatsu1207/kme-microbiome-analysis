"""
Longevity Village Analysis: Figure 1 (cohort demographics).

Panel A: age distribution of the DS and RC cohorts (density histograms,
30 bins per cohort). Panel B: sex distribution per cohort (stacked
percentages, female below male).

Usage (set paths in ml/longevity_config.py first):
    cd /path/to/kme-microbiome-analysis
    python -m ml.longevity_fig1
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ml.longevity_config import LONGEVITY_OUT
from ml.longevity_muan import load_all_data

OUT = LONGEVITY_OUT

# (female/histogram colour, male colour) per cohort
COHORT_COLORS = {"DS": ("#4878CF", "#90CAF9"), "RC": ("#D65F5F", "#EF9A9A")}


def fig1(meta):
    cohort = meta["cohort"].map({"DS": "DS", "Muan": "RC"})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: age distributions
    for name in ("DS", "RC"):
        ages = meta.loc[cohort == name, "age"]
        ax1.hist(ages, bins=30, density=True, alpha=0.6, color=COHORT_COLORS[name][0],
                 label=f"{name} (n={len(ages)})")
    ax1.set_xlabel("Age (years)")
    ax1.set_ylabel("Density")
    ax1.set_title("A. Age distribution", fontweight="bold")
    ax1.legend()

    # Panel B: sex distribution, % of all participants (as in Table 1); the
    # two RC participants without recorded sex leave a 0.2% gap at the top
    for i, name in enumerate(("DS", "RC")):
        sex = meta.loc[cohort == name, "sex"]
        female, male = 100 * (sex == "F").mean(), 100 * (sex == "M").mean()
        female_color, male_color = COHORT_COLORS[name]
        ax2.bar(i, female, color=female_color, alpha=0.7, width=0.8)
        ax2.bar(i, male, bottom=female, color=male_color, alpha=0.7, width=0.8)
        ax2.text(i, female / 2, f"Female\n{female:.1f}%", ha="center", va="center", fontsize=10)
        ax2.text(i, female + male / 2, f"Male\n{male:.1f}%", ha="center", va="center", fontsize=10)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["DS", "RC"])
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("Percentage")
    ax2.set_title("B. Sex distribution", fontweight="bold")

    plt.tight_layout()
    fig.savefig(OUT / "fig1_cohort_demographics.png", dpi=300)
    fig.savefig(OUT / "fig1_cohort_demographics.pdf")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    meta, _, _ = load_all_data()
    fig1(meta)
    print(f"  Saved fig1_cohort_demographics.pdf/.png to {OUT}")


if __name__ == "__main__":
    main()

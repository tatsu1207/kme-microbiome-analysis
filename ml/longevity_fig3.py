"""
Longevity Village Analysis: Figure 3 (key genera box plots, 5 x 3 grid).

Same 14 genera, groups, colours and q-values as fig7_key_genera in
longevity_muan.py, laid out in a 5-column grid (the published version).
q-values are RC vs DS 60+ Mann-Whitney FDR from longevity_table_s1.

Usage (set paths in ml/longevity_config.py first):
    cd /path/to/kme-microbiome-analysis
    python -m ml.longevity_fig3 [--ncols 5]
"""
import argparse
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ml.data_join import clr_transform
from ml.longevity_config import LONGEVITY_OUT
from ml.longevity_muan import COLORS, get_age_groups, load_all_data
from ml.longevity_table_s1 import table_s1

OUT = LONGEVITY_OUT

# Curated 14 genera (order as in the manuscript)
GENERA = [
    # Core lactobacilli enrichment
    "Limosilactobacillus", "Lactobacillus", "Megasphaera",
    # DS-60+ outlier pattern
    "Roseburia", "Coprococcus", "[Eubacterium] eligens group", "Haemophilus",
    # Key depletions
    "Lachnoclostridium", "Lachnospiraceae UCG-004",
    # Other enriched genera discussed in the text
    "Escherichia-Shigella", "Olsenella", "Methanobrevibacter",
    # Large CLR differences
    "Erysipelotrichaceae UCG-003", "Intestinibacter",
]


def fig3_key_genera(meta, genus_rel, da_results, ncols=5):
    groups = get_age_groups(meta)
    groups_k = groups[groups.isin(["DS young", "DS elderly", "RC elderly"])]
    genus_clr = clr_transform(genus_rel)

    nrows = -(-len(GENERA) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 3.6 * nrows))
    axes = axes.flatten()
    group_order = ["DS young", "DS elderly", "RC elderly"]
    group_colors = [COLORS["ds_young"], COLORS["ds_old"], COLORS["muan"]]

    for i, genus in enumerate(GENERA):
        ax = axes[i]
        data = [genus_clr.loc[groups_k[groups_k == g].index, genus].values for g in group_order]
        bp = ax.boxplot(data, labels=[f"DS young\n(n={len(data[0])})",
                                      f"DS older\nadults\n(n={len(data[1])})",
                                      f"RC older\nadults\n(n={len(data[2])})"],
                        patch_artist=True, widths=0.6, flierprops=dict(markersize=3))
        for patch, color in zip(bp["boxes"], group_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        # Group means as diamonds
        ax.scatter([1, 2, 3], [d.mean() for d in data], marker="D", color="black", s=20, zorder=5)

        q = da_results.loc[da_results["genus"] == genus, "q_value"].iloc[0]
        ax.set_title(f"{re.sub(r'_[0-9]+$', '', genus)}\nq = {q:.2e}", fontsize=9)
        if i % ncols == 0:
            ax.set_ylabel("CLR Abundance", fontsize=8)
        ax.tick_params(axis="x", labelsize=7)
        ax.tick_params(axis="y", labelsize=7)

    for j in range(len(GENERA), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    fig.savefig(OUT / "fig3_key_genera.pdf")
    fig.savefig(OUT / "fig3_key_genera.png", dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ncols", type=int, default=5)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    meta, genus_rel, _ = load_all_data()
    fig3_key_genera(meta, genus_rel, table_s1(meta, genus_rel), ncols=args.ncols)
    print(f"  Saved fig3_key_genera.pdf/.png to {OUT}")


if __name__ == "__main__":
    main()

"""
Longevity Village Analysis: Figure S5 (PICRUSt2 predicted functional capacity).

From the PICRUSt2 run on the ASV table rarefied to 10,000 reads:
  A  butyrate-synthesis KOs (kinase route ptb/buk/bcd; CoA-transferase route
     but A/B) for DS young, DS 60-69 and RC 60-69 — mean +/- SEM, Mann-Whitney
     DS 60-69 vs RC 60-69 (* P < 0.05, ** P < 0.01, *** P < 0.001), and fold
     change RC / DS 60-69
  B  three-group pathway classification (cohort only / both / aging only /
     neither), as in longevity_picrust2.py

Usage (set paths in ml/longevity_config.py first):
    cd /path/to/kme-microbiome-analysis
    python -m ml.longevity_figS5
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from ml.longevity_config import LONGEVITY_OUT, PICRUST2_RAREFIED_DIR
from ml.longevity_muan import load_all_data
from ml.longevity_picrust2 import (
    COA_TRANSFERASE_KOS, KINASE_KOS, load_picrust2, mw_table, prevalent,
)

OUT = LONGEVITY_OUT
GROUP_COLORS = {"ds_young": "#4878CF", "ds_6069": "#6ACC65", "rc_6069": "#D65F5F"}
GROUP_LABELS = {"ds_young": "DS young", "ds_6069": "DS 60–69", "rc_6069": "RC 60–69"}
CLASS_COLORS = {"Cohort\nonly": "#D65F5F", "Both": "#9467BD", "Aging\nonly": "#4878CF", "Neither": "lightgray"}
UP, DOWN = "#2CA02C", "#D65F5F"


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


def bracket(ax, x0, x1, y, text, italic=False):
    ax.plot([x0, x0, x1, x1], [y - 60, y, y, y - 60], color="#555555", linewidth=1.2)
    ax.text((x0 + x1) / 2, y + 40, text, ha="center", va="bottom", fontsize=10,
            color="#555555", style="italic" if italic else "normal")


def panel_a(ax, ko, groups):
    kos = {**KINASE_KOS, **COA_TRANSFERASE_KOS}
    xs = [0, 1, 2, 3.8, 4.8]
    width, offsets = 0.25, (-0.25, 0, 0.25)
    for (key, color), off in zip(GROUP_COLORS.items(), offsets):
        idx = groups[key]
        means = [ko.loc[idx, k].mean() for k in kos]
        sems = [ko.loc[idx, k].std() / np.sqrt(len(idx)) for k in kos]
        ax.bar([x + off for x in xs], means, width * 0.85, yerr=sems, capsize=3, color=color,
               alpha=0.85, label=f"{GROUP_LABELS[key]} (n={len(idx)})", error_kw={"linewidth": 1})

    ymax = max(ko.loc[groups["ds_young"], list(kos)].mean()) * 1.25
    for x, k in zip(xs, kos):
        ds, rc = ko.loc[groups["ds_6069"], k], ko.loc[groups["rc_6069"], k]
        p = stats.mannwhitneyu(rc, ds, alternative="two-sided").pvalue
        top = max(ds.mean() + ds.std() / np.sqrt(len(ds)), rc.mean() + rc.std() / np.sqrt(len(rc))) + 80
        ax.plot([x, x, x + 0.25, x + 0.25], [top - 25, top, top, top - 25], color="black", linewidth=0.8)
        ax.text(x + 0.125, top + 10, stars(p), ha="center", va="bottom", fontsize=8)
        fc = rc.mean() / ds.mean()
        ax.text(x, -ymax * 0.06, f"FC={fc:.2f}", ha="center", va="top", fontsize=9,
                fontweight="bold", color=UP if fc > 1 else DOWN)

    bracket(ax, xs[0] - 0.4, xs[2] + 0.4, ymax * 0.92, "Butyrate kinase pathway", italic=True)
    bracket(ax, xs[3] - 0.4, xs[4] + 0.4, ymax * 0.92, "CoA-transferase", italic=True)
    ax.set_xticks(xs)
    ax.set_xticklabels(list(kos.values()), fontsize=11)
    ax.tick_params(axis="x", pad=4)
    ax.set_ylim(-ymax * 0.11, ymax * 1.04)
    ax.set_ylabel("Predicted gene abundance\n(rarefied, per sample)", fontsize=11)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.88), fontsize=9)
    ax.set_title("A   Butyrate-synthesis KOs (rarefied to 10,000 reads)", loc="left",
                 fontweight="bold", fontsize=12)


def panel_b(ax, pathways, groups):
    feats = prevalent(pathways, [groups["ds_young"], groups["ds_6069"], groups["rc_6069"]])
    aging = mw_table(pathways, groups["ds_young"], groups["ds_6069"], feats)["q"] < 0.05
    cohort = (mw_table(pathways, groups["ds_6069"], groups["rc_6069"], feats)["q"] < 0.05).reindex(aging.index)
    counts = {"Cohort\nonly": int((cohort & ~aging).sum()), "Both": int((cohort & aging).sum()),
              "Aging\nonly": int((aging & ~cohort).sum()), "Neither": int((~aging & ~cohort).sum())}
    n = len(feats)
    y = np.arange(len(counts))[::-1]
    for yi, (label, c) in zip(y, counts.items()):
        ax.barh(yi, c, color=CLASS_COLORS[label], alpha=1.0 if label == "Neither" else 0.85, height=0.6)
    for yi, c in zip(y, counts.values()):
        ax.text(c + 5, yi, f"{c} ({100 * c / n:.0f}%)", va="center", fontsize=10, fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels(list(counts.keys()), fontsize=10)
    ax.set_xlim(0, max(counts.values()) * 1.22)
    ax.set_xlabel("Number of pathways", fontsize=11)
    ax.set_title(f"B   Pathway patterns\n({n}, rarefied)", loc="left", fontweight="bold", fontsize=12)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    meta, _, _ = load_all_data()
    ko, pathways, _ = load_picrust2(PICRUST2_RAREFIED_DIR)
    groups = {
        "ds_young": meta.index[(meta["cohort"] == "DS") & (meta["age"] < 40)],
        "ds_6069": meta.index[(meta["cohort"] == "DS") & meta["age"].between(60, 69)],
        "rc_6069": meta.index[(meta["cohort"] == "Muan") & meta["age"].between(60, 69)],
    }
    groups = {k: [s for s in v if s in ko.index] for k, v in groups.items()}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.8, 5.0), gridspec_kw={"width_ratios": [2.6, 1]})
    panel_a(ax1, ko, groups)
    panel_b(ax2, pathways, groups)
    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(w_pad=4)
    fig.savefig(OUT / "figS5_picrust2.png", dpi=300)
    fig.savefig(OUT / "figS5_picrust2.pdf")
    plt.close(fig)
    print(f"  Saved figS5_picrust2.pdf/.png to {OUT}")


if __name__ == "__main__":
    main()

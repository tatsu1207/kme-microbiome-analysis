"""
Longevity Village Analysis: Figure 2 (top 25 differentially abundant genera).

The 25 most significant genera (Mann-Whitney P, RC vs DS 60+) among the 70
at FDR < 0.05, ordered by CLR difference. Panel A: CLR difference
(RC - DS older adults); panel B: Cliff's delta with the negligible threshold
(|delta| = 0.147). Genera significant by both Mann-Whitney and ANCOM-BC2 but
with opposite sign are marked with a dagger; run longevity_ancombc2.py first
(without its output the daggers are omitted).

Usage (set paths in ml/longevity_config.py first):
    cd /path/to/kme-microbiome-analysis
    python -m ml.longevity_fig2
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from ml.longevity_config import LONGEVITY_OUT
from ml.longevity_muan import load_all_data
from ml.longevity_table_s1 import table_s1

OUT = LONGEVITY_OUT
N_TOP = 25
NEGLIGIBLE = 0.147
RC_COLOR, DS_COLOR = "#D9534F", "#5B8FD0"


def sign_discordant(da):
    """Genera significant by both methods with opposite direction."""
    path = OUT / "tableS3_ancombc2.csv"
    if not path.exists():
        print(f"  {path.name} not found (run longevity_ancombc2.py); daggers omitted")
        return set()
    ancom = pd.read_csv(path)
    m = da[da["q_value"] < 0.05].merge(ancom[ancom["diff"].astype(bool)], on="genus")
    return set(m.loc[np.sign(m["clr_diff"]) != np.sign(m["lfc"]), "genus"])


def fig2(da, n_rc, n_ds):
    n_sig = int((da["q_value"] < 0.05).sum())
    top = da[da["q_value"] < 0.05].nsmallest(N_TOP, "p_value").sort_values("clr_diff")
    dagger = sign_discordant(da)
    labels = [f"{g} †" if g in dagger else g for g in top["genus"]]
    colors = [RC_COLOR if d > 0 else DS_COLOR for d in top["clr_diff"]]
    y = np.arange(len(top))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 8.2))

    ax1.barh(y, top["clr_diff"], color=colors, height=0.7)
    ax1.set_yticks(y)
    ax1.set_yticklabels(labels, fontsize=10)
    ax1.axvline(0, color="black", linewidth=1)
    ax1.set_xlabel("CLR difference (RC − DS older adults)", fontsize=11)
    ax1.set_title("A. CLR Difference", fontsize=12)

    ax2.barh(y, top["cliffs_delta"], color=colors, height=0.7)
    ax2.set_yticks(y)
    ax2.set_yticklabels([])
    ax2.axvline(0, color="black", linewidth=1)
    for x in (-NEGLIGIBLE, NEGLIGIBLE):
        ax2.axvline(x, color="gray", linewidth=1, linestyle="--")
    ax2.set_xlabel("Cliff's δ (effect size)", fontsize=11)
    ax2.set_title("B. Cliff's δ", fontsize=12)

    for ax in (ax1, ax2):
        ax.set_ylim(-1, len(top))
        ax.tick_params(axis="x", labelsize=10)

    handles = [Patch(color=RC_COLOR, label="Higher in RC"),
               Patch(color=DS_COLOR, label="Higher in DS"),
               Line2D([0], [0], color="gray", linestyle="--", label=f"|δ| = {NEGLIGIBLE} (negligible)")]
    if dagger:
        handles.append(Line2D([0], [0], color="none", marker="$†$", markerfacecolor="black",
                              markeredgecolor="black", markersize=10,
                              label="† sign-discordant with ANCOM-BC2"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=11)
    fig.suptitle(f"Top {N_TOP} Differentially Abundant Genera\n"
                 f"RC (n={n_rc:,}) vs DS Older Adults (n={n_ds}), {n_sig} significant (FDR < 0.05)",
                 fontsize=14)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(OUT / "fig2_differential_abundance.png", dpi=300)
    fig.savefig(OUT / "fig2_differential_abundance.pdf")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    meta, genus_rel, _ = load_all_data()
    n_rc = int((meta["cohort"] == "Muan").sum())
    n_ds = int(((meta["cohort"] == "DS") & (meta["age"] >= 60)).sum())
    fig2(table_s1(meta, genus_rel), n_rc, n_ds)
    print(f"  Saved fig2_differential_abundance.pdf/.png to {OUT}")


if __name__ == "__main__":
    main()

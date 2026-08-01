"""
Re-run symptom-microbiome associations with age and sex as covariates.

Original analysis: Kruskal-Wallis (no covariates) per symptom x genus pair.
This script: OLS regression on CLR-transformed genus abundances:
    CLR_genus ~ C(symptom_item) + AGE + SEX
Tests the symptom item effect (F-test on indicator dummies), with per-item FDR.

Usage:
    cd /path/to/kme-microbiome-analysis
    python -m ml.run_age_sex_adjusted
"""
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multitest import multipletests

from ml.config import OUTPUT_DIR
from ml.data_join import clr_transform, load_genus_abundance, load_kmedata

warnings.filterwarnings("ignore")

OUT = OUTPUT_DIR / "results"

# The 16 items that were significant in the unadjusted analysis
ITEMS_OF_INTEREST = [
    "SCATORDSYMP03", "SCATORDSYMP22_02", "SCATORDSYMP20", "SCATORDSYMP22_05",
    "SCATORDSYMP08", "SCATORDSYMP21_05", "SCATORDSYMP22_03", "SCATORDSYMP21_07",
    "SCATORDSYMP21_02", "SCATORDSYMP10", "SCATORDSYMP15", "SCATORDSYMP11_01",
    "SCATORDSYMP12", "SCATORDSYMP17", "SCATORDSYMP16", "SCATORDSYMP19",
]

ITEM_DESCRIPTIONS = {
    "SCATORDSYMP03": "Digestion quality",
    "SCATORDSYMP08": "Sweating degree",
    "SCATORDSYMP10": "Post-sweat feeling",
    "SCATORDSYMP11_01": "No abnormal sweating",
    "SCATORDSYMP12": "Stool pattern",
    "SCATORDSYMP15": "Cold-heat preference",
    "SCATORDSYMP16": "Hand temperature",
    "SCATORDSYMP17": "Foot temperature",
    "SCATORDSYMP19": "Water intake amount",
    "SCATORDSYMP20": "Water temperature preference",
    "SCATORDSYMP21_02": "Fever tendency",
    "SCATORDSYMP21_05": "Nasal congestion",
    "SCATORDSYMP21_07": "Cough/phlegm",
    "SCATORDSYMP22_02": "Condition: digestion",
    "SCATORDSYMP22_03": "Condition: stool",
    "SCATORDSYMP22_05": "Condition: none",
}

ITEM_DOMAINS = {
    "SCATORDSYMP03": "Digestive & condition",
    "SCATORDSYMP08": "Sweating",
    "SCATORDSYMP10": "Sweating",
    "SCATORDSYMP11_01": "Sweating",
    "SCATORDSYMP12": "Digestive & condition",
    "SCATORDSYMP15": "Thermoregulation",
    "SCATORDSYMP16": "Thermoregulation",
    "SCATORDSYMP17": "Thermoregulation",
    "SCATORDSYMP19": "Thermoregulation",
    "SCATORDSYMP20": "Thermoregulation",
    "SCATORDSYMP21_02": "Cold symptoms",
    "SCATORDSYMP21_05": "Cold symptoms",
    "SCATORDSYMP21_07": "Cold symptoms",
    "SCATORDSYMP22_02": "Digestive & condition",
    "SCATORDSYMP22_03": "Digestive & condition",
    "SCATORDSYMP22_05": "Digestive & condition",
}


def load_data():
    """Load and align data."""
    kme = load_kmedata()
    genus_rel = load_genus_abundance()
    paired = kme.index.intersection(genus_rel.index)
    ds = [x for x in paired if "_DS_" in x]
    kme = kme.loc[ds]
    genus_rel = genus_rel.loc[ds]

    # CLR transform all genera (not just top 50)
    genus_clr = clr_transform(genus_rel)

    # Prevalence filter: >= 10%
    prevalent = (genus_rel > 0).mean(axis=0)
    prev_genera = prevalent[prevalent >= 0.10].index.tolist()
    print(f"Prevalent genera (>=10%): {len(prev_genera)}")

    # Meta
    meta = kme[["SEX", "AGE"]].copy()
    meta["AGE"] = pd.to_numeric(meta["AGE"], errors="coerce")
    meta["SEX_num"] = (meta["SEX"] == "F").astype(int)

    return kme, genus_clr, prev_genera, meta


def run_kruskal_unadjusted(kme, genus_clr, prev_genera, symp_col):
    """Original unadjusted Kruskal-Wallis analysis (for comparison)."""
    symp = kme[symp_col].dropna()
    common = symp.index.intersection(genus_clr.index)
    symp = symp.loc[common]
    groups_vals = sorted(symp.unique())
    if len(groups_vals) < 2:
        return pd.DataFrame()

    results = []
    for genus in prev_genera:
        gv = [genus_clr.loc[symp[symp == g].index, genus].values for g in groups_vals]
        gv = [g for g in gv if len(g) >= 5]
        if len(gv) >= 2:
            stat, p = stats.kruskal(*gv)
            results.append({"genus": genus, "p_value": p})

    if not results:
        return pd.DataFrame()
    df = pd.DataFrame(results)
    _, df["q_value"], _, _ = multipletests(df["p_value"], method="fdr_bh")
    return df


def run_adjusted(kme, genus_clr, prev_genera, meta, symp_col):
    """Age+sex adjusted analysis using OLS: CLR ~ C(symptom) + AGE + SEX."""
    symp = kme[symp_col].dropna()
    common = symp.index.intersection(genus_clr.index).intersection(meta.dropna().index)
    symp = symp.loc[common]
    groups_vals = sorted(symp.unique())
    if len(groups_vals) < 2:
        return pd.DataFrame()

    # Build design dataframe
    design = pd.DataFrame(index=common)
    design["symptom"] = symp.loc[common].astype(str)
    design["AGE"] = meta.loc[common, "AGE"]
    design["SEX_num"] = meta.loc[common, "SEX_num"]

    # Drop rows with missing age
    design = design.dropna()
    if design["symptom"].nunique() < 2:
        return pd.DataFrame()

    results = []
    for genus in prev_genera:
        design["y"] = genus_clr.loc[design.index, genus]

        # Check group sizes after filtering
        group_counts = design["symptom"].value_counts()
        valid_groups = group_counts[group_counts >= 5].index
        if len(valid_groups) < 2:
            continue

        sub = design[design["symptom"].isin(valid_groups)].copy()

        try:
            model = ols("y ~ C(symptom) + AGE + SEX_num", data=sub).fit()
            # F-test for symptom effect (Type II)
            aov = anova_lm(model, typ=2)
            p_symptom = aov.loc["C(symptom)", "PR(>F)"]
            results.append({"genus": genus, "p_value": p_symptom})
        except Exception:
            continue

    if not results:
        return pd.DataFrame()
    df = pd.DataFrame(results)
    _, df["q_value"], _, _ = multipletests(df["p_value"], method="fdr_bh")
    return df


def main():
    print("Loading data...")
    kme, genus_clr, prev_genera, meta = load_data()
    print(f"N subjects: {len(kme)}, N genera: {len(prev_genera)}")
    print(f"AGE: mean={meta['AGE'].mean():.1f}, sd={meta['AGE'].std():.1f}")
    print(f"SEX: M={sum(meta['SEX']=='M')}, F={sum(meta['SEX']=='F')}")
    print()

    comparison = []

    for item in ITEMS_OF_INTEREST:
        if item not in kme.columns:
            print(f"  {item}: NOT FOUND in data, skipping")
            continue

        # Unadjusted
        unadj = run_kruskal_unadjusted(kme, genus_clr, prev_genera, item)
        n_unadj = (unadj["q_value"] < 0.05).sum() if len(unadj) > 0 else 0

        # Adjusted
        adj = run_adjusted(kme, genus_clr, prev_genera, meta, item)
        n_adj = (adj["q_value"] < 0.05).sum() if len(adj) > 0 else 0

        top_genus_adj = ""
        top_q_adj = np.nan
        if len(adj) > 0 and n_adj > 0:
            top = adj.nsmallest(1, "q_value").iloc[0]
            top_genus_adj = top["genus"]
            top_q_adj = top["q_value"]
        elif len(adj) > 0:
            top = adj.nsmallest(1, "q_value").iloc[0]
            top_genus_adj = top["genus"]
            top_q_adj = top["q_value"]

        desc = ITEM_DESCRIPTIONS.get(item, item)
        domain = ITEM_DOMAINS.get(item, "")
        print(f"  {item} ({desc}): unadj={n_unadj}, adj={n_adj}, "
              f"top={top_genus_adj} (q={top_q_adj:.2e})")

        comparison.append({
            "symptom": item,
            "description": desc,
            "domain": domain,
            "n_genera_tested": len(adj) if len(adj) > 0 else len(unadj),
            "n_sig_unadjusted": n_unadj,
            "n_sig_adjusted": n_adj,
            "top_genus_adjusted": top_genus_adj,
            "top_q_adjusted": top_q_adj,
        })

    comp_df = pd.DataFrame(comparison)
    comp_df = comp_df.sort_values("n_sig_unadjusted", ascending=False)

    # Save comparison table
    comp_path = OUT / "age_sex_adjustment_comparison.csv"
    comp_df.to_csv(comp_path, index=False)
    print(f"\nComparison table saved to {comp_path}")
    print(comp_df.to_string(index=False))

    # Generate updated tableS1 with adjusted counts
    tableS1 = comp_df[["symptom", "description", "domain", "n_genera_tested",
                        "n_sig_adjusted", "top_genus_adjusted", "top_q_adjusted"]].copy()
    tableS1.columns = ["symptom", "description", "domain", "n_genera_tested",
                        "n_significant", "top_genus", "top_q"]
    tableS1 = tableS1.sort_values("n_significant", ascending=False)
    tableS1_path = OUT / "tableS1_symptom_microbiome_adjusted.csv"
    tableS1.to_csv(tableS1_path, index=False)
    print(f"\nUpdated tableS1 saved to {tableS1_path}")

    # Generate updated Figure 1 (symptom landscape bar chart)
    plot_df = tableS1[tableS1["n_significant"] > 0].sort_values("n_significant")
    if len(plot_df) == 0:
        plot_df = tableS1.nlargest(5, "n_significant")

    domain_colors = {
        "Digestive & condition": "#2196F3",
        "Thermoregulation": "#FF9800",
        "Sweating": "#4CAF50",
        "Cold symptoms": "#9C27B0",
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    labels = [f"{r['description']}" for _, r in plot_df.iterrows()]
    colors = [domain_colors.get(r["domain"], "#757575") for _, r in plot_df.iterrows()]
    bars = ax.barh(range(len(plot_df)), plot_df["n_significant"], color=colors)
    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Number of Significantly Associated Genera (FDR < 0.05)")
    ax.set_title("Fig. 2. Age- and Sex-Adjusted Symptom-Microbiome Associations",
                 fontweight="bold")

    # Legend for domains
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=d) for d, c in domain_colors.items()
                       if d in plot_df["domain"].values]
    if legend_elements:
        ax.legend(handles=legend_elements, loc="lower right", fontsize=8)

    plt.tight_layout()
    fig.savefig(OUT / "fig2_symptom_landscape.png", dpi=300)
    fig.savefig(OUT / "fig2_symptom_landscape.pdf")
    plt.close()
    print(f"\nFigure saved to {OUT / 'fig2_symptom_landscape.png'}")

    # Print summary statement
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    thermo_items = [r for r in comparison if r["domain"] == "Thermoregulation"]
    other_items = [r for r in comparison if r["domain"] != "Thermoregulation"]

    print("\nThermoregulation items:")
    for r in thermo_items:
        change = r["n_sig_unadjusted"] - r["n_sig_adjusted"]
        pct = (change / r["n_sig_unadjusted"] * 100) if r["n_sig_unadjusted"] > 0 else 0
        print(f"  {r['description']}: {r['n_sig_unadjusted']} -> {r['n_sig_adjusted']} "
              f"({pct:.0f}% reduction)")

    print("\nOther items:")
    for r in sorted(other_items, key=lambda x: x["n_sig_unadjusted"], reverse=True):
        change = r["n_sig_unadjusted"] - r["n_sig_adjusted"]
        pct = (change / r["n_sig_unadjusted"] * 100) if r["n_sig_unadjusted"] > 0 else 0
        print(f"  {r['description']}: {r['n_sig_unadjusted']} -> {r['n_sig_adjusted']} "
              f"({pct:.0f}% reduction)")

    total_unadj = sum(r["n_sig_unadjusted"] for r in comparison)
    total_adj = sum(r["n_sig_adjusted"] for r in comparison)
    print(f"\nTotal significant associations: {total_unadj} -> {total_adj} "
          f"({(total_unadj - total_adj) / total_unadj * 100:.0f}% reduction)")


if __name__ == "__main__":
    main()

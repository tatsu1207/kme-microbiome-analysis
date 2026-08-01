"""
Generate publication figures and tables.

Usage:
    cd /home/unnot/github/health-check/microbiome_report
    python -m ml.make_figures
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

RESULTS_DIR = Path("vault/combined/results")
ANALYSIS_DIR = Path("ml/output/analysis")
MECH_DIR = Path("ml/output/mechanisms")
ML_DIR = Path("ml/output")

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "sans-serif",
})


# ===================================================================
# FIGURE S1: ML Model Comparison (M1 vs M2 vs M3) — grouped bar chart
# ===================================================================
def figS2_model_comparison():
    print("  Figure S2: Model comparison...")

    # Load all three summaries
    dada2 = pd.read_csv(ML_DIR / "model_comparison_summary_dada2.csv")
    asv = pd.read_csv(ML_DIR / "model_comparison_summary_asv.csv")
    mothur = pd.read_csv(ML_DIR / "model_comparison_summary_mothur.csv")

    # Use dada2 genus as primary (has pathways in original run, but our "both" run used genus-only)
    df = dada2.copy()

    outcomes = df["outcome"].tolist()
    labels = [o.replace("y_", "").replace("_", " ").title() for o in outcomes]

    # Split metabolic vs digestive
    met_idx = list(range(5))
    dig_idx = list(range(5, 10))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

    for ax, idx_list, title in [
        (axes[0], met_idx, "Metabolic Outcomes"),
        (axes[1], dig_idx, "Digestive Outcomes"),
    ]:
        n = len(idx_list)
        x = np.arange(n)
        w = 0.25

        m1 = [df.iloc[i]["AUC_M1"] for i in idx_list]
        m2 = [df.iloc[i]["AUC_M2"] for i in idx_list]
        m3 = [df.iloc[i]["AUC_M3"] for i in idx_list]
        labs = [labels[i] for i in idx_list]

        bars1 = ax.bar(x - w, m1, w, label="M1: Questionnaire", color="#2196F3", alpha=0.85)
        bars2 = ax.bar(x, m2, w, label="M2: Microbiome", color="#4CAF50", alpha=0.85)
        bars3 = ax.bar(x + w, m3, w, label="M3: Combined", color="#FF9800", alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(labs, rotation=35, ha="right")
        ax.set_ylabel("AUC-ROC")
        ax.set_title(title)
        ax.set_ylim(0.35, 0.95)
        ax.legend(loc="upper right", framealpha=0.9)

        # Add significance markers for M3 vs M1
        for j, i in enumerate(idx_list):
            p = df.iloc[i].get("p_M3_vs_M1", 1.0)
            if p < 0.05:
                ax.text(x[j] + w, m3[j] + 0.01, "*", ha="center", fontsize=12)

    fig.suptitle("Fig. S2. Predictive Performance: Questionnaire vs Microbiome vs Combined",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "figS2_model_comparison.png")
    fig.savefig(RESULTS_DIR / "figS2_model_comparison.pdf")
    plt.close()


# ===================================================================
# FIGURE S3: Microbiome resolution comparison (Genus vs ASV)
# ===================================================================
def figS3_resolution_comparison():
    print("  Figure S3: Resolution comparison...")

    dada2 = pd.read_csv(ML_DIR / "model_comparison_summary_dada2.csv")
    asv = pd.read_csv(ML_DIR / "model_comparison_summary_asv.csv")

    outcomes = dada2["outcome"].tolist()
    labels = [o.replace("y_", "").replace("_", "\n") for o in outcomes]

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(outcomes))
    w = 0.3

    ax.bar(x - w/2, dada2["AUC_M2"], w, label="Genus-level (50 features)", color="#2196F3", alpha=0.85)
    ax.bar(x + w/2, asv["AUC_M2"], w, label="ASV-level (100 features)", color="#4CAF50", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("AUC-ROC (M2: Microbiome Only)")
    ax.set_title("Fig. S3. Microbiome-Only Prediction: Genus-Level vs ASV-Level",
                 fontweight="bold")
    ax.set_ylim(0.35, 0.75)
    ax.legend(loc="upper right")
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "figS3_resolution_comparison.png")
    fig.savefig(RESULTS_DIR / "figS3_resolution_comparison.pdf")
    plt.close()


# ===================================================================
# FIGURE 3: Differential Abundance — Forest plot
# ===================================================================
def fig3_differential_abundance():
    print("  Figure 3: Differential abundance...")

    da = pd.read_csv(ANALYSIS_DIR / "differential_abundance.csv")

    # Focus on metabolic outcomes with significant hits
    targets = ["y_hypertension", "y_prediabetes", "y_liver", "y_obesity"]
    target_labels = {
        "y_hypertension": "Hypertension",
        "y_prediabetes": "Prediabetes",
        "y_liver": "Liver Dysfunction",
        "y_obesity": "Obesity",
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    has_n_tools = "n_tools" in da.columns

    for ax, target in zip(axes.flat, targets):
        sub = da[(da["outcome"] == target) & (da["significant"])].copy()
        sub = sub.sort_values("clr_diff")

        if len(sub) == 0:
            ax.text(0.5, 0.5, "No significant genera", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(target_labels[target])
            continue

        top = sub

        colors = ["#E53935" if d > 0 else "#1E88E5" for d in top["clr_diff"]]
        y_pos = range(len(top))

        ax.barh(y_pos, top["clr_diff"], color=colors, alpha=0.8, height=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top["genus"], fontsize=8, fontstyle="italic")
        ax.axvline(0, color="black", linewidth=0.5)
        ax.set_xlabel("CLR Difference (disease − healthy)")
        ax.set_title(target_labels[target], fontweight="bold")

        # Annotate with tool agreement count
        if has_n_tools:
            for j, (_, row) in enumerate(top.iterrows()):
                nt = int(row["n_tools"])
                if nt > 0:
                    x_pos = row["clr_diff"]
                    offset = 0.03 if x_pos >= 0 else -0.03
                    ha = "left" if x_pos >= 0 else "right"
                    ax.text(x_pos + offset, j, f"{nt}/5",
                            va="center", ha=ha, fontsize=6.5, color="#555")

    fig.suptitle(
        "Fig. 3. Differentially Abundant Genera by Health Outcome\n"
        "(consensus: Wilcoxon + ≥3/5 tools: ALDEx2, DESeq2, ANCOM-BC2, LinDA, MaAsLin2)",
        fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "fig3_differential_abundance.png")
    fig.savefig(RESULTS_DIR / "fig3_differential_abundance.pdf")
    plt.close()


# ===================================================================
# Beta Diversity PERMANOVA (not in draft, kept for reference)
# ===================================================================
def fig2_beta_diversity():
    print("  Beta diversity (not in draft)...")

    from statsmodels.stats.multitest import multipletests
    beta = pd.read_csv(ANALYSIS_DIR / "beta_diversity_permanova.csv")
    beta["label"] = beta["outcome"].str.replace("y_", "").str.replace("_", " ").str.title()
    _, beta["q_value"], _, _ = multipletests(beta["p_value"], method="fdr_bh")
    beta = beta.sort_values("R2", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#E53935" if q < 0.05 else "#BDBDBD" for q in beta["q_value"]]

    bars = ax.barh(range(len(beta)), beta["R2"] * 100, color=colors, alpha=0.85)
    ax.set_yticks(range(len(beta)))
    ax.set_yticklabels(beta["label"])
    ax.set_xlabel("Variance Explained (R², %)")
    ax.set_title("PERMANOVA: Microbiome Composition Variance Explained by Health Status",
                 fontweight="bold")

    # Legend
    from matplotlib.patches import Patch
    ax.legend([Patch(color="#E53935"), Patch(color="#BDBDBD")],
              ["FDR q < 0.05", "FDR q ≥ 0.05"], loc="lower right")

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "fig2_beta_diversity.png")
    fig.savefig(RESULTS_DIR / "fig2_beta_diversity.pdf")
    plt.close()


# ===================================================================
# FIGURE 6: Enterotype Analysis
# ===================================================================
def fig5_enterotypes():
    print("  Figure 5: Enterotypes...")

    coords = pd.read_csv(ANALYSIS_DIR / "enterotype_pcoa_coords.csv", index_col=0)
    et_outcomes = pd.read_csv(ANALYSIS_DIR / "enterotype_outcomes.csv")

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    # Panel A: PCoA plot
    ax = axes[0]
    pct1 = coords["pct_var_PC1"].iloc[0]
    pct2 = coords["pct_var_PC2"].iloc[0]
    colors = {0: "#2196F3", 1: "#E53935"}
    labels = {0: "ET1: Prevotella", 1: "ET2: Bacteroides"}
    for et in [0, 1]:
        mask = coords["enterotype"] == et
        ax.scatter(coords.loc[mask, "PC1"], coords.loc[mask, "PC2"],
                   c=colors[et], alpha=0.4, s=15, label=labels[et])
    ax.set_xlabel(f"PCoA1 ({pct1:.1f}%)")
    ax.set_ylabel(f"PCoA2 ({pct2:.1f}%)")
    ax.set_title("A. Enterotype Clustering (PAM on JSD)", fontweight="bold")
    ax.legend()

    # Panel B: Disease prevalence by enterotype
    ax = axes[1]
    et_outcomes["label"] = et_outcomes["outcome"].str.replace("y_", "").str.replace("_", " ").str.title()

    # Parse rates
    rates_et0 = []
    rates_et1 = []
    for _, row in et_outcomes.iterrows():
        parts = row["rates"].split("|")
        r0 = float(parts[0].split("=")[1].strip().rstrip("%")) / 100
        r1 = float(parts[1].split("=")[1].strip().rstrip("%")) / 100
        rates_et0.append(r0)
        rates_et1.append(r1)

    x = np.arange(len(et_outcomes))
    w = 0.35
    ax.bar(x - w/2, rates_et0, w, label="ET1: Prevotella", color="#2196F3", alpha=0.85)
    ax.bar(x + w/2, rates_et1, w, label="ET2: Bacteroides", color="#E53935", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(et_outcomes["label"], rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Disease Prevalence")
    ax.set_title("B. Disease Prevalence by Enterotype", fontweight="bold")
    ax.legend()

    # Add significance stars (* q<0.05, ** q<0.01, *** q<0.001)
    for i, (_, row) in enumerate(et_outcomes.iterrows()):
        if row.get("significant", False):
            max_rate = max(rates_et0[i], rates_et1[i])
            q = row["q_value"]
            stars = "***" if q < 0.001 else ("**" if q < 0.01 else "*")
            ax.text(i, max_rate + 0.02, stars, ha="center", fontsize=12, fontweight="bold")

    fig.suptitle("Fig. 5. Enterotype Clustering and Disease Prevalence",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "fig5_enterotypes.png")
    fig.savefig(RESULTS_DIR / "fig5_enterotypes.pdf")
    plt.close()



# ===================================================================
# FIGURE 5: HAM/DAM ecological module balance
# ===================================================================
def fig4_ham_dam_modules():
    print("  Figure 4: HAM/DAM modules...")

    import pickle
    from scipy import stats as sp_stats
    from ml.data_join import load_genus_abundance, clr_transform

    with open(ML_DIR / "joined.pkl", "rb") as f:
        data = pickle.load(f)
    outcomes = data["outcomes"]

    genus_rel = load_genus_abundance()
    genus_rel = genus_rel.loc[outcomes.index]
    genus_clr = clr_transform(genus_rel)

    HAM = ["Oscillibacter", "Alistipes", "Akkermansia", "Odoribacter",
           "Intestinimonas", "UBA1819", "Eisenbergiella", "Blautia"]
    DAM = ["Allisonella", "Mitsuokella", "Holdemanella", "Catenibacterium", "Prevotella_9"]

    ham_score = genus_clr[HAM].mean(axis=1)
    dam_score = genus_clr[DAM].mean(axis=1)
    balance = ham_score - dam_score

    met_outcomes = ["y_obesity", "y_hypertension", "y_prediabetes", "y_liver"]
    met_labels = ["Obesity", "Hypertension", "Prediabetes", "Liver\nDysfunction"]

    fig, axes = plt.subplots(1, 4, figsize=(14, 4.5), sharey=True)

    for ax, outcome, label in zip(axes, met_outcomes, met_labels):
        y = outcomes[outcome].dropna()
        h_vals = balance.loc[y[y == 0].index]
        d_vals = balance.loc[y[y == 1].index]

        bp = ax.boxplot([h_vals, d_vals], positions=[0, 1], widths=0.6,
                        patch_artist=True, showfliers=False,
                        medianprops=dict(color="black", linewidth=1.5))
        bp["boxes"][0].set_facecolor("#4CAF50")
        bp["boxes"][0].set_alpha(0.7)
        bp["boxes"][1].set_facecolor("#E53935")
        bp["boxes"][1].set_alpha(0.7)

        for pos, vals, color in [(0, h_vals, "#4CAF50"), (1, d_vals, "#E53935")]:
            jitter = np.random.normal(0, 0.06, len(vals))
            ax.scatter(pos + jitter, vals, alpha=0.15, s=8, color=color, zorder=2)

        stat, p = sp_stats.mannwhitneyu(h_vals, d_vals, alternative="two-sided")
        stars = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
        ymax = max(h_vals.quantile(0.95), d_vals.quantile(0.95))
        ax.text(0.5, ymax + 0.8, stars, ha="center", fontsize=14, fontweight="bold")
        p_str = f"p = {p:.1e}" if p < 0.001 else f"p = {p:.3f}"
        ax.text(0.5, ymax + 0.35, p_str, ha="center", fontsize=7, color="#555")

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Healthy", "Disease"], fontsize=9)
        ax.set_title(label, fontweight="bold")

    axes[0].set_ylabel("Module Balance Score\n(HAM − DAM)", fontsize=11)

    fig.suptitle("Fig. 4. Ecological Module Balance by Health Status",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "fig4_ham_dam_modules.png", dpi=300, bbox_inches="tight")
    fig.savefig(RESULTS_DIR / "fig4_ham_dam_modules.pdf", bbox_inches="tight")
    plt.close()


# ===================================================================
# FIGURE S1: Cold-Heat Constitution
# ===================================================================
def figS5_cold_heat():
    print("  Figure S5: Cold-heat constitution...")

    km_path = ANALYSIS_DIR / "km_cold_heat_genera.csv"
    if not km_path.exists():
        print("    Skipping (no KM data)")
        return

    km = pd.read_csv(km_path)
    km_sig = km[km["significant"]].sort_values("kruskal_H", ascending=False)

    if len(km_sig) == 0:
        # Show top 10 by p-value even if not significant
        km_sig = km.nsmallest(10, "p_value")

    fig, ax = plt.subplots(figsize=(8, 5))

    colors = ["#E53935" if d > 0 else "#1E88E5" for d in km_sig["clr_hot_minus_cold"]]
    ax.barh(range(len(km_sig)), km_sig["clr_hot_minus_cold"], color=colors, alpha=0.8)
    ax.set_yticks(range(len(km_sig)))
    ax.set_yticklabels(km_sig["genus"], fontsize=9)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("CLR Difference (Hot − Cold constitution)")
    ax.set_title("Fig. S5. Genera Associated with Cold-Heat Constitution",
                 fontweight="bold")

    for i, (_, row) in enumerate(km_sig.iterrows()):
        marker = "*" if row.get("significant", False) else ""
        if marker:
            ax.text(row["clr_hot_minus_cold"] + 0.02 * np.sign(row["clr_hot_minus_cold"]),
                    i, marker, va="center", fontsize=12)

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "figS5_cold_heat.png")
    fig.savefig(RESULTS_DIR / "figS5_cold_heat.pdf")
    plt.close()


# ===================================================================
# FIGURE 2: Symptom-Microbiome Association Landscape (age/sex-adjusted)
# ===================================================================
def fig2_symptom_landscape():
    print("  Figure 2: Symptom landscape...")

    comp = pd.read_csv(RESULTS_DIR / "age_sex_adjustment_comparison.csv")
    plot_df = comp[comp["n_sig_adjusted"] > 0].sort_values("n_sig_adjusted")
    if len(plot_df) == 0:
        plot_df = comp.nlargest(5, "n_sig_adjusted")

    domain_colors = {
        "Digestive & condition": "#2196F3",
        "Thermoregulation": "#FF9800",
        "Sweating": "#4CAF50",
        "Cold symptoms": "#9C27B0",
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    labels = [r["description"] for _, r in plot_df.iterrows()]
    colors = [domain_colors.get(r["domain"], "#757575") for _, r in plot_df.iterrows()]
    ax.barh(range(len(plot_df)), plot_df["n_sig_adjusted"], color=colors)
    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Number of Significantly Associated Genera (FDR < 0.05)")
    ax.set_title("Fig. 2. Age- and Sex-Adjusted Symptom-Microbiome Associations",
                 fontweight="bold")

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=d) for d, c in domain_colors.items()
                       if d in plot_df["domain"].values]
    if legend_elements:
        ax.legend(handles=legend_elements, loc="lower right", fontsize=8)

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "fig2_symptom_landscape.png", dpi=300)
    fig.savefig(RESULTS_DIR / "fig2_symptom_landscape.pdf")
    plt.close()


# ===================================================================
# FIGURE S1: Age and Sex Effects on Gut Microbiota Composition
# ===================================================================
def figS1_age_sex_microbiome():
    print("  Figure S1: Age and sex effects...")

    age_df = pd.read_csv(RESULTS_DIR / "age_genus_associations.csv")
    sex_df = pd.read_csv(RESULTS_DIR / "sex_genus_associations.csv")

    # Top 20 by absolute rho (age) / by significance i.e. q-value (sex)
    age_top = age_df.reindex(age_df["rho"].abs().nlargest(20).index).sort_values("rho")
    sex_top = sex_df.nsmallest(20, "q_value").sort_values("mean_diff")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: Age
    ax = axes[0]
    colors_a = ["#2196F3" if r > 0 else "#E53935" for r in age_top["rho"]]
    ax.barh(range(len(age_top)), age_top["rho"], color=colors_a, alpha=0.85)
    ax.set_yticks(range(len(age_top)))
    ax.set_yticklabels(age_top["genus"], fontsize=8)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Spearman rho with age")
    ax.set_title("A. Top 20 age-associated genera", fontweight="bold")
    from matplotlib.patches import Patch
    ax.legend([Patch(color="#2196F3"), Patch(color="#E53935")],
              ["Increase with age", "Decrease with age"], loc="lower right", fontsize=8)

    # Panel B: Sex
    ax = axes[1]
    colors_s = ["#9C27B0" if d > 0 else "#FF9800" for d in sex_top["mean_diff"]]
    ax.barh(range(len(sex_top)), sex_top["mean_diff"], color=colors_s, alpha=0.85)
    ax.set_yticks(range(len(sex_top)))
    ax.set_yticklabels(sex_top["genus"], fontsize=8)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Mean CLR difference (F - M)")
    ax.set_title("B. Top 20 sex-associated genera", fontweight="bold")
    ax.legend([Patch(color="#9C27B0"), Patch(color="#FF9800")],
              ["Higher in F", "Higher in M"], loc="lower left", fontsize=8)

    fig.suptitle("Fig. S1. Age and Sex Effects on Gut Microbiota Composition (N=774)",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "figS1_age_sex_microbiome.png", dpi=300)
    fig.savefig(RESULTS_DIR / "figS1_age_sex_microbiome.pdf")
    plt.close()


# ===================================================================
# FIGURE S4: Effect of Age/Sex Adjustment on Associations
# ===================================================================
def figS4_adjustment_comparison():
    print("  Figure S4: Adjustment comparison...")

    comp = pd.read_csv(RESULTS_DIR / "age_sex_adjustment_comparison.csv")
    comp = comp.sort_values("n_sig_unadjusted", ascending=False)
    # Only items with any signal in either analysis
    comp = comp[(comp["n_sig_unadjusted"] > 0) | (comp["n_sig_adjusted"] > 0)]

    domain_colors = {
        "Digestive & condition": "#2196F3",
        "Thermoregulation": "#FF9800",
        "Sweating": "#4CAF50",
        "Cold symptoms": "#9C27B0",
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    # Panel A: Unadjusted
    ax = axes[0]
    colors = [domain_colors.get(r["domain"], "#757575") for _, r in comp.iterrows()]
    ax.barh(range(len(comp)), comp["n_sig_unadjusted"], color=colors)
    ax.set_yticks(range(len(comp)))
    ax.set_yticklabels(comp["description"], fontsize=8)
    ax.set_xlabel("Significant genera (FDR < 0.05)")
    ax.set_title("Unadjusted (Kruskal-Wallis)", fontweight="bold")
    ax.invert_yaxis()

    # Panel B: Adjusted
    ax = axes[1]
    ax.barh(range(len(comp)), comp["n_sig_adjusted"], color=colors)
    ax.set_xlabel("Significant genera (FDR < 0.05)")
    ax.set_title("Age + Sex Adjusted (OLS)", fontweight="bold")
    ax.invert_yaxis()

    # Legend
    from matplotlib.patches import Patch
    used_domains = comp["domain"].unique()
    legend_elements = [Patch(facecolor=domain_colors.get(d, "#757575"), label=d)
                       for d in domain_colors if d in used_domains]
    fig.legend(handles=legend_elements, loc="lower center", ncol=len(legend_elements),
               fontsize=9, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("Fig. S4. Effect of Age and Sex Adjustment on Symptom-Microbiome Associations",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "figS4_adjustment_comparison.png", dpi=300)
    fig.savefig(RESULTS_DIR / "figS4_adjustment_comparison.pdf")
    plt.close()


# ===================================================================
# TABLES
# ===================================================================
def make_tables():
    print("\n  Creating tables...")

    # Table 1: Model comparison summary
    dada2 = pd.read_csv(ML_DIR / "model_comparison_summary_dada2.csv")
    dada2["Outcome"] = dada2["outcome"].str.replace("y_", "").str.replace("_", " ").str.title()
    table1 = dada2[["Outcome", "AUC_M1", "AUC_M2", "AUC_M3",
                     "delta_AUC_M3_vs_M1", "p_M3_vs_M1"]].copy()
    table1.columns = ["Outcome", "M1 (Quest.)", "M2 (Microbiome)", "M3 (Combined)",
                       "ΔAUC (M3-M1)", "p (DeLong)"]
    for col in table1.columns[1:]:
        table1[col] = table1[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "")
    table1.to_csv(RESULTS_DIR / "table1_model_comparison.csv", index=False)
    print("    Table 1: Model comparison")

    # Table 2: PERMANOVA
    beta = pd.read_csv(ANALYSIS_DIR / "beta_diversity_permanova.csv")
    beta["Outcome"] = beta["outcome"].str.replace("y_", "").str.replace("_", " ").str.title()
    table2 = beta[["Outcome", "pseudo_F", "R2", "p_value", "n_samples"]].copy()
    table2.columns = ["Outcome", "Pseudo-F", "R² (%)", "p-value", "n"]
    table2["Pseudo-F"] = table2["Pseudo-F"].apply(lambda x: f"{x:.3f}")
    table2["R² (%)"] = table2["R² (%)"].apply(lambda x: f"{x * 100:.2f}")
    table2["p-value"] = table2["p-value"].apply(lambda x: f"{x:.4f}")
    table2.to_csv(RESULTS_DIR / "table2_permanova.csv", index=False)
    print("    Table 2: PERMANOVA")

    # Table S2: Differential abundance — significant genera
    da = pd.read_csv(ANALYSIS_DIR / "differential_abundance.csv")
    da_sig = da[da["significant"]].copy()
    da_sig["Outcome"] = da_sig["outcome"].str.replace("y_", "").str.replace("_", " ").str.title()
    tableS2 = da_sig[["Outcome", "genus", "clr_diff", "q_value", "direction"]].copy()
    tableS2.columns = ["Outcome", "Genus", "CLR Difference", "FDR q-value", "Direction"]
    tableS2["CLR Difference"] = tableS2["CLR Difference"].apply(lambda x: f"{x:+.3f}")
    tableS2["FDR q-value"] = tableS2["FDR q-value"].apply(lambda x: f"{x:.4f}")
    tableS2 = tableS2.sort_values(["Outcome", "FDR q-value"])
    tableS2.to_csv(RESULTS_DIR / "tableS2_differential_abundance.csv", index=False)
    print("    Table S2: Differential abundance")



# ===================================================================
# MAIN
# ===================================================================
def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating figures...")
    fig2_symptom_landscape()
    fig3_differential_abundance()
    fig4_ham_dam_modules()
    fig5_enterotypes()
    figS1_age_sex_microbiome()
    figS2_model_comparison()
    figS3_resolution_comparison()
    figS4_adjustment_comparison()
    figS5_cold_heat()

    make_tables()

    print(f"\nAll figures and tables saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()

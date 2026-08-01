"""
Mechanistic analysis: Taxa → Metabolic Pathways → Health Outcomes

1. Taxa-pathway correlation: which genera drive which pathways?
2. Mediation analysis: does a pathway mediate taxon→outcome effects?
3. KEGG Ortholog enrichment: specific functional genes in disease groups
4. Mechanistic summary: integrated taxa→pathway→outcome chains

Usage:
    cd /path/to/kme-microbiome-analysis
    python -m ml.mechanisms
"""
import gzip
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from ml.config import DADA2_DIR, OUTPUT_DIR, TAXONOMY
from ml.data_join import (
    clr_transform,
    compute_outcomes,
    load_genus_abundance,
    load_kmedata,
    load_pathways,
)

warnings.filterwarnings("ignore")

MECH_DIR = OUTPUT_DIR / "mechanisms"


# ===================================================================
# 1. TAXA-PATHWAY CORRELATION
# ===================================================================
def taxa_pathway_correlation(genus_clr, pw_clr, sig_genera, sig_pathways, output_dir):
    """
    Compute Spearman correlations between significant genera and
    significant pathways. Identifies which taxa drive which functions.
    """
    print("\n" + "=" * 70)
    print("1. TAXA-PATHWAY CORRELATION NETWORK")
    print("=" * 70)

    # Align indices
    common = genus_clr.index.intersection(pw_clr.index)
    g = genus_clr.loc[common]
    p = pw_clr.loc[common]

    # Focus on significant genera and pathways
    gen_cols = [c for c in g.columns if c in sig_genera]
    pw_cols = [c for c in p.columns if c in sig_pathways]

    print(f"  Testing {len(gen_cols)} genera × {len(pw_cols)} pathways")

    results = []
    for genus in gen_cols:
        for pathway in pw_cols:
            rho, pval = stats.spearmanr(g[genus].values, p[pathway].values)
            results.append({
                "genus": genus,
                "pathway": pathway,
                "spearman_rho": rho,
                "p_value": pval,
            })

    df = pd.DataFrame(results)
    if len(df) == 0:
        print("  No pairs to test.")
        return df

    _, df["q_value"], _, _ = multipletests(df["p_value"], method="fdr_bh")
    df["significant"] = df["q_value"] < 0.05
    df = df.sort_values("p_value")

    n_sig = df["significant"].sum()
    print(f"  {n_sig} significant genus-pathway correlations (FDR < 0.05)")

    # Show strongest correlations
    sig_df = df[df["significant"] & (df["spearman_rho"].abs() >= 0.15)].copy()
    sig_df = sig_df.sort_values("spearman_rho", key=abs, ascending=False)
    print(f"\n  Strong correlations (|rho| >= 0.15, FDR < 0.05):")
    for _, row in sig_df.head(30).iterrows():
        direction = "+" if row["spearman_rho"] > 0 else "-"
        print(f"    {row['genus']:<30s} {direction} {row['pathway']:<25s} "
              f"rho={row['spearman_rho']:+.3f}  q={row['q_value']:.4f}")

    df.to_csv(output_dir / "taxa_pathway_correlations.csv", index=False)
    return df


# ===================================================================
# 2. MEDIATION ANALYSIS
# ===================================================================
def mediation_analysis(genus_clr, pw_clr, outcomes, sig_genera, sig_pathways,
                       output_dir):
    """
    For significant (genus, pathway, outcome) triples:
    Test if pathway mediates the genus → outcome relationship.

    Baron & Kenny steps:
      a: genus → pathway (linear regression)
      b: pathway → outcome (logistic, controlling for genus)
      c: genus → outcome (total effect)
      c': genus → outcome controlling for pathway (direct effect)
      Indirect effect = c - c' (Sobel test)
    """
    print("\n" + "=" * 70)
    print("2. MEDIATION ANALYSIS: Taxa → Pathway → Outcome")
    print("=" * 70)

    from statsmodels.api import OLS, Logit, add_constant

    common = genus_clr.index.intersection(pw_clr.index).intersection(outcomes.index)

    binary_outcomes = ["y_obesity", "y_hypertension", "y_prediabetes", "y_liver"]
    gen_cols = [c for c in genus_clr.columns if c in sig_genera]
    pw_cols = [c for c in pw_clr.columns if c in sig_pathways]

    all_results = []
    tested = 0

    for target in binary_outcomes:
        y = outcomes.loc[common, target].dropna()
        valid_idx = y.index

        if y.sum() < 20 or (1 - y).sum() < 20:
            continue

        for genus in gen_cols:
            g_vals = genus_clr.loc[valid_idx, genus]

            # Step c: total effect (genus → outcome)
            X_c = add_constant(g_vals)
            try:
                model_c = Logit(y, X_c).fit(disp=0)
                coef_c = model_c.params.iloc[1]
                p_c = model_c.pvalues.iloc[1]
            except Exception:
                continue

            if p_c > 0.05:
                continue  # No total effect to mediate

            for pathway in pw_cols:
                p_vals = pw_clr.loc[valid_idx, pathway]

                # Step a: genus → pathway (OLS)
                X_a = add_constant(g_vals)
                try:
                    model_a = OLS(p_vals, X_a).fit()
                    coef_a = model_a.params.iloc[1]
                    p_a = model_a.pvalues.iloc[1]
                except Exception:
                    continue

                if p_a > 0.05:
                    continue  # No a-path

                # Step b+c': outcome ~ genus + pathway
                X_bc = add_constant(pd.DataFrame({
                    "genus": g_vals, "pathway": p_vals
                }))
                try:
                    model_bc = Logit(y, X_bc).fit(disp=0)
                    coef_b = model_bc.params["pathway"]
                    p_b = model_bc.pvalues["pathway"]
                    coef_cprime = model_bc.params["genus"]
                    p_cprime = model_bc.pvalues["genus"]
                except Exception:
                    continue

                # Sobel test for indirect effect
                se_a = model_a.bse.iloc[1]
                se_b = model_bc.bse["pathway"]
                sobel_se = np.sqrt(coef_a**2 * se_b**2 + coef_b**2 * se_a**2)
                indirect = coef_a * coef_b
                if sobel_se > 0:
                    sobel_z = indirect / sobel_se
                    sobel_p = 2 * stats.norm.sf(abs(sobel_z))
                else:
                    sobel_z = 0
                    sobel_p = 1.0

                # Proportion mediated
                if abs(coef_c) > 1e-10:
                    prop_mediated = 1 - (coef_cprime / coef_c)
                else:
                    prop_mediated = 0

                tested += 1
                all_results.append({
                    "outcome": target,
                    "genus": genus,
                    "pathway": pathway,
                    "coef_total_c": coef_c,
                    "p_total_c": p_c,
                    "coef_a_genus_to_pw": coef_a,
                    "p_a": p_a,
                    "coef_b_pw_to_outcome": coef_b,
                    "p_b": p_b,
                    "coef_direct_cprime": coef_cprime,
                    "p_cprime": p_cprime,
                    "indirect_effect": indirect,
                    "sobel_z": sobel_z,
                    "sobel_p": sobel_p,
                    "proportion_mediated": prop_mediated,
                })

    print(f"  Tested {tested} (genus, pathway, outcome) triples")

    if not all_results:
        print("  No significant mediations found.")
        return pd.DataFrame()

    med_df = pd.DataFrame(all_results)
    _, med_df["sobel_q"], _, _ = multipletests(med_df["sobel_p"], method="fdr_bh")
    med_df = med_df.sort_values("sobel_p")

    sig_med = med_df[med_df["sobel_q"] < 0.10]  # Relaxed threshold for mediation
    print(f"  {len(sig_med)} significant mediations (Sobel FDR < 0.10)")

    if len(sig_med) > 0:
        print(f"\n  {'Outcome':<18s} {'Genus':<25s} → {'Pathway':<22s} → Outcome  "
              f"{'%Med':>6s} {'Sobel q':>8s}")
        print("  " + "-" * 90)
        for _, row in sig_med.head(20).iterrows():
            print(f"  {row['outcome']:<18s} {row['genus']:<25s} → "
                  f"{row['pathway']:<22s} → Outcome  "
                  f"{row['proportion_mediated']:>5.1%} {row['sobel_q']:>8.4f}")

    # Also show marginal results
    marginal = med_df[(med_df["sobel_p"] < 0.05) & (med_df["sobel_q"] >= 0.10)]
    if len(marginal) > 0:
        print(f"\n  Nominally significant mediations (p < 0.05, FDR >= 0.10):")
        for _, row in marginal.head(15).iterrows():
            print(f"  {row['outcome']:<18s} {row['genus']:<25s} → "
                  f"{row['pathway']:<22s}  "
                  f"%med={row['proportion_mediated']:.1%}  p={row['sobel_p']:.4f}")

    med_df.to_csv(output_dir / "mediation_results.csv", index=False)
    return med_df


# ===================================================================
# 3. KEGG ORTHOLOG ENRICHMENT
# ===================================================================
def ko_enrichment(outcomes, output_dir):
    """
    Test individual KEGG orthologs for differential abundance.
    More granular than MetaCyc pathways — identifies specific enzymes.
    """
    print("\n" + "=" * 70)
    print("3. KEGG ORTHOLOG (KO) ENRICHMENT")
    print("=" * 70)

    ko_path = DADA2_DIR / "picrust2/KO_metagenome_out/pred_metagenome_unstrat.tsv.gz"
    ko_raw = pd.read_csv(ko_path, sep="\t", index_col=0).T  # samples × KOs

    # Load described version for names
    ko_desc_path = DADA2_DIR / "picrust2/KO_metagenome_out/pred_metagenome_unstrat_described.tsv.gz"
    ko_names = {}
    if ko_desc_path.exists():
        ko_desc = pd.read_csv(ko_desc_path, sep="\t", index_col=0, nrows=0)
        desc_col = ko_desc.columns[0] if len(ko_desc.columns) > 0 else None

        # Re-read with descriptions
        ko_desc_full = pd.read_csv(ko_desc_path, sep="\t", nrows=0)
        for col in ko_desc_full.columns[1:]:  # skip 'function' or 'description'
            pass
        # The row index has "KO: description" format
        ko_desc_idx = pd.read_csv(ko_desc_path, sep="\t", usecols=[0])
        for val in ko_desc_idx.iloc[:, 0]:
            if ":" in str(val):
                parts = str(val).split(":", 1)
                ko_names[parts[0].strip()] = parts[1].strip()

    # CLR transform
    ko_clr = clr_transform(ko_raw + 1e-6)

    # Filter to prevalent KOs (present in >50% samples)
    prevalence = (ko_raw > 0).mean(axis=0)
    prevalent = prevalence[prevalence >= 0.5].index.tolist()
    print(f"  Testing {len(prevalent)} KOs with >=50% prevalence (out of {ko_raw.shape[1]})")

    metabolic_targets = ["y_obesity", "y_hypertension", "y_prediabetes", "y_liver"]

    # Key KEGG pathways of interest for health
    # We'll test all prevalent KOs but highlight those in key pathways
    scfa_kos = {
        "K00929": "butyrate kinase",
        "K01034": "acetate CoA-transferase alpha",
        "K01035": "acetate CoA-transferase beta",
        "K00248": "butyryl-CoA dehydrogenase",
        "K01571": "oxaloacetate decarboxylase",
        "K00626": "acetyl-CoA C-acetyltransferase",
    }
    bile_kos = {
        "K01442": "bile salt hydrolase (choloylglycine hydrolase)",
        "K07007": "7-alpha-hydroxysteroid dehydrogenase",
    }
    lps_kos = {
        "K02535": "lipid A biosynthesis (lpxC)",
        "K00677": "lipid A biosynthesis (lpxA)",
        "K02536": "lipid A biosynthesis (lpxD)",
    }
    tma_kos = {
        "K07811": "choline TMA-lyase (cutC)",
        "K07821": "TMA-lyase activating enzyme (cutD)",
    }

    all_results = []
    for target in metabolic_targets:
        y = outcomes[target].dropna()
        common = ko_clr.index.intersection(y.index)
        y = y.loc[common]
        X = ko_clr.loc[common, prevalent]

        pos_idx = y[y == 1].index
        neg_idx = y[y == 0].index

        pvals = []
        effects = []
        kos = []
        for ko in prevalent:
            stat, p = stats.mannwhitneyu(
                X.loc[pos_idx, ko].values,
                X.loc[neg_idx, ko].values,
                alternative="two-sided"
            )
            pvals.append(p)
            effects.append(X.loc[pos_idx, ko].mean() - X.loc[neg_idx, ko].mean())
            kos.append(ko)

        reject, qvals, _, _ = multipletests(pvals, method="fdr_bh")

        df = pd.DataFrame({
            "ko": kos,
            "description": [ko_names.get(k, "") for k in kos],
            "clr_diff": effects,
            "p_value": pvals,
            "q_value": qvals,
            "significant": reject,
            "outcome": target,
        }).sort_values("p_value")

        n_sig = df["significant"].sum()
        print(f"\n  {target}: {n_sig} significant KOs (FDR < 0.05)")

        # Show top significant KOs
        if n_sig > 0:
            for _, row in df[df["significant"]].head(10).iterrows():
                desc = row["description"][:45] if row["description"] else row["ko"]
                direction = "↑disease" if row["clr_diff"] > 0 else "↓disease"
                print(f"    {row['ko']:<10s} {desc:<48s} {direction}  "
                      f"q={row['q_value']:.4f}")

        # Highlight health-relevant KOs regardless of significance
        print(f"\n    Health-relevant KOs for {target}:")
        for ko_dict, label in [(scfa_kos, "SCFA"), (bile_kos, "Bile acid"),
                                (lps_kos, "LPS"), (tma_kos, "TMA/TMAO")]:
            for ko, name in ko_dict.items():
                if ko in df["ko"].values:
                    row = df[df["ko"] == ko].iloc[0]
                    marker = " *" if row["significant"] else ""
                    direction = "↑" if row["clr_diff"] > 0 else "↓"
                    print(f"      [{label}] {ko} {name}: "
                          f"{direction} CLR={row['clr_diff']:+.3f}  "
                          f"p={row['p_value']:.4f}  q={row['q_value']:.4f}{marker}")

        all_results.append(df)

    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv(output_dir / "ko_enrichment.csv", index=False)
    print(f"\n  Saved KO enrichment results.")
    return combined


# ===================================================================
# 4. MECHANISTIC SUMMARY
# ===================================================================
def mechanistic_summary(taxa_pw_df, mediation_df, ko_df, da_df, output_dir):
    """
    Integrate all results into mechanistic chains:
    Taxon → (correlated pathways) → (KO genes) → Health outcome
    """
    print("\n" + "=" * 70)
    print("4. INTEGRATED MECHANISTIC SUMMARY")
    print("=" * 70)

    # Load differential abundance results
    if da_df is None or len(da_df) == 0:
        print("  No differential abundance data.")
        return

    sig_da = da_df[da_df["significant"]].copy()

    # For each outcome, build the chain
    for outcome in sig_da["outcome"].unique():
        print(f"\n  === {outcome} ===")

        # Significant genera for this outcome
        outcome_genera = sig_da[sig_da["outcome"] == outcome].sort_values("q_value")

        for _, genus_row in outcome_genera.head(8).iterrows():
            genus = genus_row["genus"]
            direction = "↑" if genus_row["clr_diff"] > 0 else "↓"

            print(f"\n    {direction} {genus} (CLR diff={genus_row['clr_diff']:+.3f}, "
                  f"q={genus_row['q_value']:.4f})")

            # Correlated pathways
            if taxa_pw_df is not None and len(taxa_pw_df) > 0:
                genus_pw = taxa_pw_df[
                    (taxa_pw_df["genus"] == genus) &
                    (taxa_pw_df["significant"]) &
                    (taxa_pw_df["spearman_rho"].abs() >= 0.10)
                ].sort_values("spearman_rho", key=abs, ascending=False)

                if len(genus_pw) > 0:
                    print(f"      Correlated pathways:")
                    for _, pw_row in genus_pw.head(5).iterrows():
                        pw_dir = "+" if pw_row["spearman_rho"] > 0 else "-"
                        print(f"        {pw_dir} {pw_row['pathway']:<25s} "
                              f"rho={pw_row['spearman_rho']:+.3f}")

            # Mediation results
            if mediation_df is not None and len(mediation_df) > 0:
                genus_med = mediation_df[
                    (mediation_df["genus"] == genus) &
                    (mediation_df["outcome"] == outcome) &
                    (mediation_df["sobel_p"] < 0.05)
                ]
                if len(genus_med) > 0:
                    print(f"      Mediated through:")
                    for _, med_row in genus_med.head(3).iterrows():
                        print(f"        → {med_row['pathway']:<25s} "
                              f"({med_row['proportion_mediated']:.0%} mediated, "
                              f"Sobel p={med_row['sobel_p']:.4f})")

    # Cross-outcome summary
    print(f"\n  === CROSS-OUTCOME MECHANISMS ===")
    multi_outcome = sig_da.groupby("genus").agg(
        n_outcomes=("outcome", "nunique"),
        outcomes=("outcome", lambda x: ", ".join(sorted(x))),
        mean_effect=("clr_diff", "mean"),
    ).query("n_outcomes >= 2").sort_values("n_outcomes", ascending=False)

    for genus, row in multi_outcome.iterrows():
        direction = "↑" if row["mean_effect"] > 0 else "↓"
        print(f"\n    {direction} {genus} ({row['n_outcomes']} outcomes: {row['outcomes']})")

        # Show correlated pathways for this genus
        if taxa_pw_df is not None and len(taxa_pw_df) > 0:
            genus_pw = taxa_pw_df[
                (taxa_pw_df["genus"] == genus) &
                (taxa_pw_df["significant"]) &
                (taxa_pw_df["spearman_rho"].abs() >= 0.15)
            ].sort_values("spearman_rho", key=abs, ascending=False)
            if len(genus_pw) > 0:
                for _, pw_row in genus_pw.head(3).iterrows():
                    pw_dir = "+" if pw_row["spearman_rho"] > 0 else "-"
                    print(f"      {pw_dir} {pw_row['pathway']} (rho={pw_row['spearman_rho']:+.3f})")

    # Save integrated summary
    summary_path = output_dir / "mechanistic_summary.txt"
    print(f"\n  All results saved to {output_dir}")


# ===================================================================
# MAIN
# ===================================================================
def main():
    MECH_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    kme = load_kmedata()
    genus_rel = load_genus_abundance()
    pw_raw = load_pathways()

    # Align to paired DS subjects
    paired = kme.index.intersection(genus_rel.index)
    ds_mask = pd.Index(paired).str.contains("_DS_")
    paired = paired[ds_mask]

    kme = kme.loc[paired]
    genus_rel = genus_rel.loc[paired]
    outcomes = compute_outcomes(kme)

    genus_clr = clr_transform(genus_rel)
    pw_clr = clr_transform(pw_raw.loc[paired] + 1e-6)

    print(f"Working with {len(paired)} DS subjects")

    # Load differential abundance results from previous analysis
    da_path = OUTPUT_DIR / "analysis" / "differential_abundance.csv"
    da_df = pd.read_csv(da_path) if da_path.exists() else None

    # Get sets of significant genera and pathways
    sig_genera = set()
    if da_df is not None:
        sig_genera = set(da_df[da_df["significant"]]["genus"].unique())
    print(f"Significant genera from DA: {len(sig_genera)}")

    pw_enrich_path = OUTPUT_DIR / "analysis" / "pathway_enrichment.csv"
    sig_pathways = set()
    if pw_enrich_path.exists():
        pw_df = pd.read_csv(pw_enrich_path)
        sig_pathways = set(pw_df[pw_df["significant"]]["pathway"].unique())
    print(f"Significant pathways from enrichment: {len(sig_pathways)}")

    # If no significant results, use top by p-value
    if len(sig_genera) == 0:
        print("  Warning: no significant genera, using top 20 by min p-value")
        sig_genera = set(da_df.nsmallest(20, "p_value")["genus"].unique())
    if len(sig_pathways) == 0:
        print("  Warning: no significant pathways, using top 20")
        if pw_enrich_path.exists():
            sig_pathways = set(pw_df.nsmallest(20, "p_value")["pathway"].unique())

    # Run analyses
    taxa_pw_df = taxa_pathway_correlation(
        genus_clr, pw_clr, sig_genera, sig_pathways, MECH_DIR
    )

    mediation_df = mediation_analysis(
        genus_clr, pw_clr, outcomes, sig_genera, sig_pathways, MECH_DIR
    )

    ko_df = ko_enrichment(outcomes, MECH_DIR)

    mechanistic_summary(taxa_pw_df, mediation_df, ko_df, da_df, MECH_DIR)

    print("\n" + "=" * 70)
    print("MECHANISTIC ANALYSIS COMPLETE")
    print(f"Results saved to: {MECH_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()

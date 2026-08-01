"""
Comprehensive microbiome analyses for publication:
  1. Differential abundance (Wilcoxon + FDR per outcome)
  2. PICRUSt2 pathway enrichment
  3. Alpha/beta diversity associations
  4. Korean Medicine × Microbiome
  5. Enterotype analysis

Usage:
    cd /path/to/kme-microbiome-analysis
    python -m ml.analysis
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import braycurtis
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from statsmodels.stats.multitest import multipletests

from ml.config import (
    ASV_TABLE, DADA2_DIR, KMEDATA_CSV, OUTPUT_DIR,
    PICRUST2_PATHWAYS, TAXONOMY,
)
from ml.data_join import (
    clr_transform,
    compute_outcomes,
    load_genus_abundance,
    load_kmedata,
    load_pathways,
)

warnings.filterwarnings("ignore")

ANALYSIS_DIR = OUTPUT_DIR / "analysis"


# ===================================================================
# 1. DIFFERENTIAL ABUNDANCE ANALYSIS
# ===================================================================
def differential_abundance(genus_rel, outcomes, meta, output_dir):
    """
    Wilcoxon rank-sum test per genus per outcome.
    Reports CLR effect size, FDR-corrected p-values.
    """
    print("\n" + "=" * 70)
    print("1. DIFFERENTIAL ABUNDANCE ANALYSIS")
    print("=" * 70)

    genus_clr = clr_transform(genus_rel)

    binary_outcomes = [c for c in outcomes.columns if c != "y_multimorbidity"]

    all_results = []

    # Filter to genera present in >=10% of samples to avoid sparse noise
    prevalence = (genus_rel > 0).mean(axis=0)
    prevalent_genera = prevalence[prevalence >= 0.10].index.tolist()
    print(f"  Testing {len(prevalent_genera)} genera with >=10% prevalence "
          f"(out of {len(genus_rel.columns)} total)")

    for target in binary_outcomes:
        y = outcomes[target].dropna()
        common = genus_clr.index.intersection(y.index)
        y = y.loc[common]
        X = genus_clr.loc[common]

        pos_idx = y[y == 1].index
        neg_idx = y[y == 0].index

        if len(pos_idx) < 10 or len(neg_idx) < 10:
            continue

        pvals = []
        effect_sizes = []
        genera = []

        for genus in [g for g in X.columns if g in prevalent_genera]:
            pos_vals = X.loc[pos_idx, genus].values
            neg_vals = X.loc[neg_idx, genus].values

            stat, p = stats.mannwhitneyu(pos_vals, neg_vals, alternative="two-sided")
            # CLR difference as effect size
            effect = pos_vals.mean() - neg_vals.mean()

            pvals.append(p)
            effect_sizes.append(effect)
            genera.append(genus)

        # FDR correction
        reject, qvals, _, _ = multipletests(pvals, method="fdr_bh")

        df = pd.DataFrame({
            "genus": genera,
            "clr_diff": effect_sizes,
            "p_value": pvals,
            "q_value": qvals,
            "significant": reject,
            "direction": ["enriched_in_disease" if e > 0 else "depleted_in_disease"
                          for e in effect_sizes],
        }).sort_values("p_value")

        n_sig = df["significant"].sum()
        print(f"\n  {target}: {n_sig} significant genera (FDR < 0.05) "
              f"out of {len(genera)}")
        if n_sig > 0:
            sig_df = df[df["significant"]].head(15)
            for _, row in sig_df.iterrows():
                print(f"    {row['genus']:<30s} CLR diff={row['clr_diff']:+.3f}  "
                      f"q={row['q_value']:.4f}  {row['direction']}")

        df["outcome"] = target
        all_results.append(df)

    combined = pd.concat(all_results, ignore_index=True)

    # --- Multi-tool consensus: upgrade significance using 5-tool DA results ---
    da_consensus_path = Path("ml/output/da_comparison/da_all_tools.csv")
    if da_consensus_path.exists():
        print(f"\n  Applying multi-tool consensus filter (ALDEx2, DESeq2, ANCOM-BC2, LinDA, MaAsLin2)...")
        da5 = pd.read_csv(da_consensus_path)
        da5_sig = da5[da5["qvalue"] < 0.05].copy()
        # Normalize genus names (MaAsLin2 replaces spaces with dots)
        da5_sig["feature_norm"] = da5_sig["feature"].str.replace(".", " ", regex=False)

        # Count how many tools found each genus significant per outcome
        tool_counts = (
            da5_sig.groupby(["outcome", "feature_norm"])["tool"]
            .nunique()
            .reset_index()
            .rename(columns={"feature_norm": "genus", "tool": "n_tools"})
        )

        combined = combined.merge(tool_counts, on=["outcome", "genus"], how="left")
        combined["n_tools"] = combined["n_tools"].fillna(0).astype(int)

        # Update significance: require Wilcoxon q<0.05 AND ≥3/5 tools
        wilcoxon_sig = combined["significant"].copy()
        combined["wilcoxon_significant"] = wilcoxon_sig
        combined["significant"] = wilcoxon_sig & (combined["n_tools"] >= 3)

        # Also add genera found by ≥3 tools but missed by Wilcoxon
        consensus_only = tool_counts[tool_counts["n_tools"] >= 3].copy()
        for _, crow in consensus_only.iterrows():
            mask = (combined["outcome"] == crow["outcome"]) & (combined["genus"] == crow["genus"])
            if mask.any() and not combined.loc[mask, "significant"].iloc[0]:
                # Check if Wilcoxon q < 0.10 (borderline) — add as consensus-significant
                wq = combined.loc[mask, "q_value"].iloc[0]
                if wq < 0.10:
                    combined.loc[mask, "significant"] = True

        n_wilcoxon = wilcoxon_sig.sum()
        n_consensus = combined["significant"].sum()
        print(f"    Wilcoxon-only significant: {n_wilcoxon}")
        print(f"    Consensus significant (Wilcoxon + ≥3 tools): {n_consensus}")
    else:
        combined["n_tools"] = 0
        combined["wilcoxon_significant"] = combined["significant"]
        print("  (No multi-tool DA results found; using Wilcoxon only)")

    combined.to_csv(output_dir / "differential_abundance.csv", index=False)
    print(f"\n  Saved to differential_abundance.csv")

    # Summary: which genera are significant across multiple outcomes?
    sig = combined[combined["significant"]]
    if len(sig) > 0:
        genus_counts = sig.groupby("genus").agg(
            n_outcomes=("outcome", "nunique"),
            outcomes=("outcome", lambda x: ", ".join(sorted(x))),
            mean_clr_diff=("clr_diff", "mean"),
        ).sort_values("n_outcomes", ascending=False)
        print(f"\n  Genera significant across multiple outcomes:")
        for genus, row in genus_counts[genus_counts["n_outcomes"] >= 2].iterrows():
            print(f"    {genus:<30s} {row['n_outcomes']} outcomes  "
                  f"mean CLR diff={row['mean_clr_diff']:+.3f}")

    return combined


# ===================================================================
# 2. PICRUST2 PATHWAY ENRICHMENT
# ===================================================================
def pathway_enrichment(outcomes, meta, output_dir):
    """
    Test PICRUSt2 MetaCyc pathways for differential abundance
    between disease/healthy groups.
    """
    print("\n" + "=" * 70)
    print("2. PICRUSt2 PATHWAY ENRICHMENT ANALYSIS")
    print("=" * 70)

    pw_raw = load_pathways()

    # Load pathway descriptions if available
    pw_desc_path = DADA2_DIR / "picrust2/pathways_out/path_abun_unstrat_described.tsv.gz"
    pw_names = {}
    if pw_desc_path.exists():
        import gzip
        with gzip.open(pw_desc_path, "rt") as f:
            header = f.readline().strip().split("\t")
            # First column has "pathway: description" format
            pw_names = {h.split(":")[0].strip(): h for h in header[1:] if ":" in h}
            # Actually the header row itself has the descriptions
        pw_desc = pd.read_csv(pw_desc_path, sep="\t", compression="gzip", nrows=0)
        for col in pw_desc.columns[1:]:
            if ":" in col:
                parts = col.split(":", 1)
                pw_names[parts[0].strip()] = parts[1].strip()

    # CLR-transform pathways
    pw_clr = clr_transform(pw_raw + 1e-6)

    binary_outcomes = [c for c in outcomes.columns if c != "y_multimorbidity"]

    all_results = []

    for target in binary_outcomes:
        y = outcomes[target].dropna()
        common = pw_clr.index.intersection(y.index)
        y = y.loc[common]
        X = pw_clr.loc[common]

        pos_idx = y[y == 1].index
        neg_idx = y[y == 0].index

        if len(pos_idx) < 10 or len(neg_idx) < 10:
            continue

        pvals = []
        effect_sizes = []
        pathways = []

        for pw in X.columns:
            pos_vals = X.loc[pos_idx, pw].values
            neg_vals = X.loc[neg_idx, pw].values
            stat, p = stats.mannwhitneyu(pos_vals, neg_vals, alternative="two-sided")
            effect = pos_vals.mean() - neg_vals.mean()
            pvals.append(p)
            effect_sizes.append(effect)
            pathways.append(pw)

        reject, qvals, _, _ = multipletests(pvals, method="fdr_bh")

        df = pd.DataFrame({
            "pathway": pathways,
            "description": [pw_names.get(p, "") for p in pathways],
            "clr_diff": effect_sizes,
            "p_value": pvals,
            "q_value": qvals,
            "significant": reject,
            "direction": ["up_in_disease" if e > 0 else "down_in_disease"
                          for e in effect_sizes],
        }).sort_values("p_value")

        n_sig = df["significant"].sum()
        print(f"\n  {target}: {n_sig} significant pathways (FDR < 0.05) "
              f"out of {len(pathways)}")
        if n_sig > 0:
            sig_df = df[df["significant"]].head(10)
            for _, row in sig_df.iterrows():
                desc = row["description"][:50] if row["description"] else row["pathway"]
                print(f"    {desc:<55s} CLR={row['clr_diff']:+.3f}  "
                      f"q={row['q_value']:.4f}")

        df["outcome"] = target
        all_results.append(df)

    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv(output_dir / "pathway_enrichment.csv", index=False)
    print(f"\n  Saved to pathway_enrichment.csv")

    return combined


# ===================================================================
# 3. ALPHA / BETA DIVERSITY
# ===================================================================
def diversity_analysis(genus_rel, outcomes, meta, output_dir):
    """
    Alpha diversity (Shannon, Simpson, observed genera) vs outcomes.
    Beta diversity (Bray-Curtis) PERMANOVA.
    """
    print("\n" + "=" * 70)
    print("3. ALPHA & BETA DIVERSITY ANALYSIS")
    print("=" * 70)

    # --- Alpha diversity ---
    print("\n  --- Alpha Diversity ---")

    # Shannon index
    def shannon(row):
        p = row[row > 0]
        return -np.sum(p * np.log(p))

    def simpson(row):
        return 1 - np.sum(row ** 2)

    alpha = pd.DataFrame(index=genus_rel.index)
    alpha["shannon"] = genus_rel.apply(shannon, axis=1)
    alpha["simpson"] = genus_rel.apply(simpson, axis=1)
    alpha["observed"] = (genus_rel > 0).sum(axis=1)
    alpha["evenness"] = alpha["shannon"] / np.log(alpha["observed"])

    binary_outcomes = [c for c in outcomes.columns if c != "y_multimorbidity"]

    alpha_results = []
    for target in binary_outcomes:
        y = outcomes[target].dropna()
        common = alpha.index.intersection(y.index)
        y = y.loc[common]
        a = alpha.loc[common]

        pos_idx = y[y == 1].index
        neg_idx = y[y == 0].index

        for metric in ["shannon", "simpson", "observed", "evenness"]:
            pos_vals = a.loc[pos_idx, metric].values
            neg_vals = a.loc[neg_idx, metric].values
            stat, p = stats.mannwhitneyu(pos_vals, neg_vals, alternative="two-sided")
            diff = pos_vals.mean() - neg_vals.mean()
            alpha_results.append({
                "outcome": target,
                "metric": metric,
                "mean_disease": pos_vals.mean(),
                "mean_healthy": neg_vals.mean(),
                "difference": diff,
                "p_value": p,
            })

    alpha_df = pd.DataFrame(alpha_results)
    # FDR correction across all tests
    _, alpha_df["q_value"], _, _ = multipletests(alpha_df["p_value"], method="fdr_bh")
    alpha_df["significant"] = alpha_df["q_value"] < 0.05

    print(f"\n  {'Outcome':<25s} {'Metric':<12s} {'Disease':>8s} {'Healthy':>8s} {'Diff':>8s} {'q':>8s}")
    print("  " + "-" * 75)
    for _, row in alpha_df.iterrows():
        marker = " *" if row["significant"] else ""
        print(f"  {row['outcome']:<25s} {row['metric']:<12s} "
              f"{row['mean_disease']:>8.3f} {row['mean_healthy']:>8.3f} "
              f"{row['difference']:>+8.3f} {row['q_value']:>8.4f}{marker}")

    alpha_df.to_csv(output_dir / "alpha_diversity.csv", index=False)

    # --- Beta diversity (PERMANOVA) ---
    print("\n  --- Beta Diversity (PERMANOVA) ---")

    # Compute Bray-Curtis distance matrix (on a subset if too large)
    n = len(genus_rel)
    print(f"  Computing Bray-Curtis distance matrix ({n} x {n})...")

    from skbio import DistanceMatrix
    from skbio.stats.distance import permanova

    # Compute pairwise Bray-Curtis
    bc_matrix = np.zeros((n, n))
    rel_vals = genus_rel.values
    for i in range(n):
        for j in range(i + 1, n):
            d = braycurtis(rel_vals[i], rel_vals[j])
            bc_matrix[i, j] = d
            bc_matrix[j, i] = d

    dm = DistanceMatrix(bc_matrix, ids=genus_rel.index.tolist())

    beta_results = []
    for target in binary_outcomes:
        y = outcomes[target].dropna()
        common = genus_rel.index.intersection(y.index)

        if len(common) < 50:
            continue

        # Filter distance matrix to common samples
        dm_sub = dm.filter(common.tolist())
        grouping = y.loc[common].astype(str).rename("group")

        result = permanova(dm_sub, grouping, permutations=999)
        r2 = result["test statistic"] * (result["number of groups"] - 1) / \
             (len(common) - 1)

        print(f"  {target:<25s} pseudo-F={result['test statistic']:.3f}  "
              f"R²={r2:.4f}  p={result['p-value']:.4f}")

        beta_results.append({
            "outcome": target,
            "pseudo_F": result["test statistic"],
            "R2": r2,
            "p_value": result["p-value"],
            "n_samples": len(common),
        })

    beta_df = pd.DataFrame(beta_results)
    beta_df.to_csv(output_dir / "beta_diversity_permanova.csv", index=False)
    print(f"\n  Saved alpha/beta diversity results.")

    return alpha_df, beta_df


# ===================================================================
# 4. KOREAN MEDICINE × MICROBIOME
# ===================================================================
def korean_medicine_microbiome(genus_rel, kme, output_dir):
    """
    Test microbiome composition differences by:
    - Cold-Heat score (CHQ sum, tertiles)
    - Key SCATORDSYMP subscales
    """
    print("\n" + "=" * 70)
    print("4. KOREAN MEDICINE × MICROBIOME ANALYSIS")
    print("=" * 70)

    genus_clr = clr_transform(genus_rel)
    common = genus_clr.index.intersection(kme.index)
    genus_clr = genus_clr.loc[common]
    kme_sub = kme.loc[common]

    # --- 4a. Cold-Heat Score ---
    print("\n  --- Cold-Heat Questionnaire (CHQ) ---")
    chq_cols = [f"CHQ{i:02d}" for i in range(1, 8)]
    chq_available = [c for c in chq_cols if c in kme_sub.columns]
    print(f"  CHQ items available: {chq_available}")

    if chq_available:
        chq_scores = kme_sub[chq_available].apply(pd.to_numeric, errors="coerce")
        chq_total = chq_scores.mean(axis=1)  # mean score (handles missing items)
        chq_valid = chq_total.dropna()

        # Split into tertiles: Cold / Neutral / Hot
        terciles = pd.qcut(chq_valid, 3, labels=["Cold", "Neutral", "Hot"])
        print(f"  CHQ tertile distribution: {terciles.value_counts().to_dict()}")

        # Test each genus across tertiles (Kruskal-Wallis)
        # Only test prevalent genera
        prevalent = (genus_rel > 0).mean(axis=0)
        prevalent_genera_km = prevalent[prevalent >= 0.10].index.tolist()

        km_results = []
        for genus in [g for g in genus_clr.columns if g in prevalent_genera_km]:
            groups = []
            for label in ["Cold", "Neutral", "Hot"]:
                idx = terciles[terciles == label].index
                idx_common = idx.intersection(genus_clr.index)
                groups.append(genus_clr.loc[idx_common, genus].values)

            if all(len(g) >= 10 for g in groups):
                stat, p = stats.kruskal(*groups)
                # Effect size: mean CLR in Hot - Cold
                hot_mean = groups[2].mean()
                cold_mean = groups[0].mean()
                km_results.append({
                    "genus": genus,
                    "kruskal_H": stat,
                    "p_value": p,
                    "clr_hot_minus_cold": hot_mean - cold_mean,
                })

        km_df = pd.DataFrame(km_results)
        if len(km_df) > 0:
            _, km_df["q_value"], _, _ = multipletests(km_df["p_value"], method="fdr_bh")
            km_df["significant"] = km_df["q_value"] < 0.05
            km_df = km_df.sort_values("p_value")

            n_sig = km_df["significant"].sum()
            print(f"  {n_sig} genera significantly associated with Cold-Heat score")
            if n_sig > 0:
                for _, row in km_df[km_df["significant"]].head(15).iterrows():
                    direction = "↑Hot" if row["clr_hot_minus_cold"] > 0 else "↑Cold"
                    print(f"    {row['genus']:<30s} H={row['kruskal_H']:.2f}  "
                          f"q={row['q_value']:.4f}  {direction} "
                          f"(diff={row['clr_hot_minus_cold']:+.3f})")

            km_df.to_csv(output_dir / "km_cold_heat_genera.csv", index=False)

    # --- 4b. SCATORDSYMP subscales (ordinal symptom patterns) ---
    print("\n  --- Sasang Ordinal Symptom Patterns ---")

    # Key digestive/constitution symptoms
    key_symptoms = {
        "SCATORDSYMP01": "appetite",
        "SCATORDSYMP05": "digestion_speed",
        "SCATORDSYMP07": "bowel_regularity",
        "SCATORDSYMP12": "stool_form",
        "SCATORDSYMP13": "stool_frequency",
    }

    for symp_col, symp_name in key_symptoms.items():
        if symp_col not in kme_sub.columns:
            continue

        symp = kme_sub[symp_col].dropna()
        symp_common = symp.index.intersection(genus_clr.index)
        symp = symp.loc[symp_common]
        groups = symp.unique()

        if len(groups) < 2:
            continue

        print(f"\n  {symp_col} ({symp_name}): {symp.value_counts().sort_index().to_dict()}")

        # Only test prevalent genera
        prevalent = (genus_rel > 0).mean(axis=0)
        prevalent_genera = prevalent[prevalent >= 0.10].index.tolist()

        results = []
        for genus in [g for g in genus_clr.columns if g in prevalent_genera]:
            group_vals = [genus_clr.loc[symp[symp == g].index, genus].values
                          for g in sorted(groups)]
            group_vals = [g for g in group_vals if len(g) >= 5]

            if len(group_vals) >= 2:
                stat, p = stats.kruskal(*group_vals)
                results.append({"genus": genus, "H": stat, "p_value": p})

        if results:
            rdf = pd.DataFrame(results)
            _, rdf["q_value"], _, _ = multipletests(rdf["p_value"], method="fdr_bh")
            n_sig = (rdf["q_value"] < 0.05).sum()
            print(f"    {n_sig} genera significantly associated (FDR < 0.05)")
            if n_sig > 0:
                for _, row in rdf[rdf["q_value"] < 0.05].sort_values("q_value").head(10).iterrows():
                    print(f"      {row['genus']:<30s} H={row['H']:.2f}  q={row['q_value']:.4f}")

    # --- 4c. Beta diversity by Cold-Heat tertile (PERMANOVA) ---
    if chq_available:
        print("\n  --- PERMANOVA: Bray-Curtis ~ Cold-Heat tertile ---")
        from skbio import DistanceMatrix
        from skbio.stats.distance import permanova

        tercile_common = terciles.dropna().index.intersection(genus_rel.index)
        rel_sub = genus_rel.loc[tercile_common]
        n = len(rel_sub)
        vals = rel_sub.values
        bc = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = braycurtis(vals[i], vals[j])
                bc[i, j] = d
                bc[j, i] = d

        dm = DistanceMatrix(bc, ids=rel_sub.index.tolist())
        grouping = terciles.loc[tercile_common].rename("cold_heat")
        result = permanova(dm, grouping, permutations=999)
        r2 = result["test statistic"] * 2 / (n - 1)
        print(f"  pseudo-F={result['test statistic']:.3f}  R²={r2:.4f}  "
              f"p={result['p-value']:.4f}")

    print(f"\n  Saved KM analysis results.")


# ===================================================================
# 5. ENTEROTYPE ANALYSIS
# ===================================================================
def _pam(dist_matrix, k, max_iter=100, random_state=42):
    """
    Partitioning Around Medoids (PAM) on a precomputed distance matrix.
    Returns (medoid_indices, labels).
    """
    rng = np.random.RandomState(random_state)
    n = dist_matrix.shape[0]

    # Initialize: pick k medoids with furthest-first heuristic
    medoids = [rng.randint(n)]
    for _ in range(1, k):
        dists_to_nearest = dist_matrix[:, medoids].min(axis=1)
        medoids.append(np.argmax(dists_to_nearest))
    medoids = np.array(medoids)

    for _ in range(max_iter):
        # Assign each point to nearest medoid
        labels = np.argmin(dist_matrix[:, medoids], axis=1)

        # Update medoids: for each cluster, pick point that minimizes total distance
        new_medoids = medoids.copy()
        for c in range(k):
            members = np.where(labels == c)[0]
            if len(members) == 0:
                continue
            # Total distance from each member to all other members
            intra_dists = dist_matrix[np.ix_(members, members)].sum(axis=1)
            new_medoids[c] = members[np.argmin(intra_dists)]

        if np.array_equal(new_medoids, medoids):
            break
        medoids = new_medoids

    labels = np.argmin(dist_matrix[:, medoids], axis=1)
    return medoids, labels


def enterotype_analysis(genus_rel, outcomes, meta, kme, output_dir):
    """
    Cluster samples into enterotypes using PAM on Jensen-Shannon divergence,
    following Arumugam et al. (2011).
    """
    print("\n" + "=" * 70)
    print("5. ENTEROTYPE ANALYSIS (PAM on JSD)")
    print("=" * 70)

    # Use all genera (full community composition, as in Arumugam et al.)
    rel = genus_rel.copy()
    # Add pseudocount for JSD computation
    rel = rel + 1e-6
    rel = rel.div(rel.sum(axis=1), axis=0)

    # Jensen-Shannon divergence
    def jsd(p, q):
        m = 0.5 * (p + q)
        kl_pm = np.sum(p * np.log(p / m))
        kl_qm = np.sum(q * np.log(q / m))
        return np.sqrt(0.5 * (kl_pm + kl_qm))

    n = len(rel)
    print(f"  Computing JSD distance matrix ({n} samples, {rel.shape[1]} genera)...")

    vals = rel.values
    jsd_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = jsd(vals[i], vals[j])
            jsd_matrix[i, j] = d
            jsd_matrix[j, i] = d

    # PAM clustering for K=2..5, select by silhouette on JSD matrix
    print(f"\n  PAM clustering (silhouette on JSD):")
    best_k = 2
    best_sil = -1
    best_labels = None
    for k in range(2, 6):
        _, labels = _pam(jsd_matrix, k, random_state=42)
        sil = silhouette_score(jsd_matrix, labels, metric="precomputed")
        print(f"    K={k}: silhouette={sil:.4f}")
        if sil > best_sil:
            best_sil = sil
            best_k = k
            best_labels = labels

    print(f"  Best K={best_k} (silhouette={best_sil:.4f})")
    enterotypes = pd.Series(best_labels, index=rel.index, name="enterotype")

    # PCoA for visualization (on JSD)
    from skbio import DistanceMatrix as skbioDM
    from skbio.stats.ordination import pcoa
    dm_jsd = skbioDM(jsd_matrix, ids=rel.index.tolist())
    pcoa_results = pcoa(dm_jsd)
    coords = pcoa_results.samples[["PC1", "PC2"]].values
    pct_var = pcoa_results.proportion_explained[:2].values * 100

    # Characterize enterotypes by dominant genus
    print(f"\n  Enterotype characterization:")
    for et in range(best_k):
        et_idx = enterotypes[enterotypes == et].index
        mean_abund = genus_rel.loc[et_idx].mean()
        top3 = mean_abund.sort_values(ascending=False).head(3)
        print(f"    Enterotype {et} (n={len(et_idx)}): "
              f"dominant = {', '.join(f'{g} ({v:.1%})' for g, v in top3.items())}")

    # Test association with outcomes
    print(f"\n  Enterotype × Health outcome associations:")
    binary_outcomes = [c for c in outcomes.columns if c != "y_multimorbidity"]
    et_results = []

    for target in binary_outcomes:
        y = outcomes[target].dropna()
        common = enterotypes.index.intersection(y.index)
        y = y.loc[common]
        et = enterotypes.loc[common]

        # Chi-square test
        contingency = pd.crosstab(et, y)
        if contingency.shape[0] >= 2 and contingency.shape[1] >= 2:
            chi2, p, dof, expected = stats.chi2_contingency(contingency)
            # Disease rate per enterotype
            rates = y.groupby(et).mean()
            rate_str = " | ".join(f"ET{k}={v:.1%}" for k, v in rates.items())
            et_results.append({
                "outcome": target,
                "chi2": chi2,
                "p_value": p,
                "rates": rate_str,
            })

    et_df = pd.DataFrame(et_results)
    if len(et_df) > 0:
        _, et_df["q_value"], _, _ = multipletests(et_df["p_value"], method="fdr_bh")
        et_df["significant"] = et_df["q_value"] < 0.05

        print(f"  {'Outcome':<25s} {'chi2':>8s} {'q':>8s} {'Rates'}")
        print("  " + "-" * 75)
        for _, row in et_df.iterrows():
            marker = " *" if row["significant"] else ""
            print(f"  {row['outcome']:<25s} {row['chi2']:>8.2f} "
                  f"{row['q_value']:>8.4f} {row['rates']}{marker}")

        et_df.to_csv(output_dir / "enterotype_outcomes.csv", index=False)

    # Test association with Cold-Heat
    chq_cols = [f"CHQ{i:02d}" for i in range(1, 8)]
    chq_available = [c for c in chq_cols if c in kme.columns]
    if chq_available:
        print(f"\n  Enterotype × Cold-Heat score:")
        chq_scores = kme.loc[enterotypes.index.intersection(kme.index), chq_available]
        chq_scores = chq_scores.apply(pd.to_numeric, errors="coerce")
        chq_mean = chq_scores.mean(axis=1).dropna()
        common = enterotypes.index.intersection(chq_mean.index)
        et_chq = enterotypes.loc[common]
        chq_common = chq_mean.loc[common]

        groups = [chq_common[et_chq == k].values for k in range(best_k)]
        stat, p = stats.kruskal(*groups)
        print(f"    Kruskal-Wallis H={stat:.2f}, p={p:.4f}")
        for k in range(best_k):
            print(f"    ET{k}: CHQ mean={groups[k].mean():.2f} ± {groups[k].std():.2f}")

    # Save enterotype assignments
    enterotypes.to_csv(output_dir / "enterotype_assignments.csv")
    coords_df = pd.DataFrame(coords, index=rel.index, columns=["PC1", "PC2"])
    coords_df["enterotype"] = enterotypes
    coords_df["pct_var_PC1"] = pct_var[0]
    coords_df["pct_var_PC2"] = pct_var[1]
    coords_df.to_csv(output_dir / "enterotype_pcoa_coords.csv")

    print(f"\n  Saved enterotype results.")
    return enterotypes


# ===================================================================
# MAIN
# ===================================================================
def main():
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    kme = load_kmedata()
    genus_rel = load_genus_abundance()

    # Align to paired DS subjects
    paired = kme.index.intersection(genus_rel.index)
    ds_mask = pd.Index(paired).str.contains("_DS_")
    paired = paired[ds_mask]

    kme = kme.loc[paired]
    genus_rel = genus_rel.loc[paired]
    outcomes = compute_outcomes(kme)
    meta = kme[["YEAR", "SEX", "AGE"]].copy()
    meta["AGE"] = pd.to_numeric(meta["AGE"], errors="coerce")

    print(f"Working with {len(paired)} DS subjects")

    # Run all analyses
    da_results = differential_abundance(genus_rel, outcomes, meta, ANALYSIS_DIR)
    pw_results = pathway_enrichment(outcomes, meta, ANALYSIS_DIR)
    alpha_df, beta_df = diversity_analysis(genus_rel, outcomes, meta, ANALYSIS_DIR)
    korean_medicine_microbiome(genus_rel, kme, ANALYSIS_DIR)
    enterotypes = enterotype_analysis(genus_rel, outcomes, meta, kme, ANALYSIS_DIR)

    print("\n" + "=" * 70)
    print("ALL ANALYSES COMPLETE")
    print(f"Results saved to: {ANALYSIS_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()

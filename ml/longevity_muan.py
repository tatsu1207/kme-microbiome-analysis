"""
Longevity Village Analysis: Muan-gun vs DS general population.

Compares the gut microbiome of elderly residents of ten designated longevity
villages in Jeollanam-do (rural cohort, RC; "Muan"/"MB" in code) with the DS
health-checkup cohort spanning ages 20-85.

Usage (set paths in ml/longevity_config.py first):
    cd /path/to/kme-microbiome-analysis
    python -m ml.longevity_muan
"""
import json
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

from ml.config import RANDOM_STATE
from ml.data_join import clr_transform, load_genus_abundance, load_kmedata, select_top_genera
from ml.longevity_config import (
    LONGEVITY_ASV_TABLE, LONGEVITY_OUT, LONGEVITY_TAXONOMY, MB_META, MB_NAMES,
    MB_QUESTIONNAIRE_DIR,
)

warnings.filterwarnings("ignore")

OUT = LONGEVITY_OUT

COLORS = {
    "muan": "#E53935",
    "ds_old": "#1E88E5",
    "ds_young": "#4CAF50",
    "muan_light": "#EF9A9A",
    "ds_light": "#90CAF9",
}

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 11,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.family": "sans-serif",
})


# ===================================================================
# DATA LOADING
# ===================================================================

def load_mb_metadata():
    """Load Muan-gun metadata (sex, height, weight, birthdate)."""
    mb = pd.read_csv(MB_META, sep="\t")
    mb = mb.rename(columns={"Sample": "sample_id", "Sex": "sex",
                            "Height": "height", "Weight": "weight",
                            "Birthdate": "birthdate"})
    mb["birthdate"] = pd.to_datetime(mb["birthdate"], format="%Y%m%d", errors="coerce")
    # Compute age at time of sampling
    year_map = {22: 2022, 23: 2023, 24: 2024, 25: 2025}
    mb["sample_year"] = mb["run_year"].map(year_map)
    mb["age"] = mb["sample_year"] - mb["birthdate"].dt.year
    mb["bmi"] = mb["weight"] / (mb["height"] ** 2)
    mb = mb.set_index("sample_id")
    return mb


def deduplicate_mb(mb_meta, genus_rel):
    """Remove repeat subjects across MB batches.

    Uses Name+Birthdate for 22_MB (from MB.xlsx), and Birthdate+Sex for
    cross-batch matching of 24_MB/25_MB (no name data available).
    Keeps the most recent sample when duplicates are found.
    """
    # Only keep samples that have microbiome data
    common = mb_meta.index.intersection(genus_rel.index)
    mb = mb_meta.loc[common].copy()

    # Load 22_MB names for precise matching
    if MB_NAMES.exists():
        names_df = pd.read_excel(MB_NAMES)
        names_df["sample_id"] = names_df["Sample"]
        names_df = names_df.set_index("sample_id")
        # For 22_MB: use Name+Birthdate (handles same-name-different-person)
        mb["name"] = names_df.reindex(mb.index)["Name"]

    # Build dedup key:
    # - 22_MB with name: Name + Birthdate (unique per person)
    # - 24_MB/25_MB without name: Birthdate + Sex (best available)
    has_name = mb["name"].notna() if "name" in mb.columns else pd.Series(False, index=mb.index)
    mb["dedup_key"] = ""
    mb.loc[has_name, "dedup_key"] = (
        mb.loc[has_name, "name"].astype(str) + "_" +
        mb.loc[has_name, "birthdate"].astype(str)
    )
    mb.loc[~has_name, "dedup_key"] = (
        mb.loc[~has_name, "birthdate"].astype(str) + "_" +
        mb.loc[~has_name, "sex"].astype(str)
    )

    # Cross-batch matching: for 24_MB/25_MB, check if Birthdate+Sex matches
    # a 22_MB subject (who has a name). If so, merge under that name key.
    if has_name.any():
        name_lookup = mb.loc[has_name, ["name", "birthdate", "sex"]].copy()
        name_lookup["match_key"] = (
            name_lookup["birthdate"].astype(str) + "_" +
            name_lookup["sex"].astype(str)
        )
        match_dict = dict(zip(name_lookup["match_key"], name_lookup.index))

        for idx in mb.index[~has_name]:
            bday_sex = mb.loc[idx, "birthdate"].isoformat()[:10] + "_" + str(mb.loc[idx, "sex"])
            # Also try the format from name_lookup
            key_try = str(mb.loc[idx, "birthdate"]) + "_" + str(mb.loc[idx, "sex"])
            if key_try in match_dict:
                matched_22 = match_dict[key_try]
                mb.loc[idx, "dedup_key"] = mb.loc[matched_22, "dedup_key"]

    n_before = len(mb)
    # Keep the most recent sample per person
    mb = mb.sort_values("run_year", ascending=False)
    mb = mb[~mb["dedup_key"].duplicated(keep="first")]
    mb = mb.drop(columns=["dedup_key", "name"], errors="ignore")

    print(f"  MB after deduplication: {len(mb)} (removed {n_before - len(mb)} "
          f"repeats from {n_before} with microbiome)")
    return mb


def load_mb_questionnaire():
    """Load MB questionnaire data (health status, lifestyle) across years.

    Returns DataFrame indexed by sample_id (e.g. 22_MB_001) with columns:
    hypertension, diabetes, osteoarthritis, smoking, exercise_days.
    """
    all_rows = []

    for year, fname in [(22, "22.xlsx"), (23, "23.xlsx"),
                        (24, "24.xlsx"), (25, "25.xlsx")]:
        fpath = MB_QUESTIONNAIRE_DIR / fname
        if not fpath.exists():
            continue

        # Health status sheet
        xl = pd.ExcelFile(fpath)
        health_sheets = [s for s in xl.sheet_names if "건강상태" in s or "건강" in s]
        lifestyle_sheets = [s for s in xl.sheet_names if "건강생활" in s]

        # Parse health status
        health_df = None
        for hs in health_sheets:
            df = pd.read_excel(fpath, sheet_name=hs)
            if "연구대상자ID" not in df.columns:
                continue
            # Find HTN and DM columns
            htn_cols = [c for c in df.columns if "고혈압" in c and "약" not in c and "관련" not in c]
            dm_cols = [c for c in df.columns if "당뇨" in c and "약" not in c]
            oa_cols = [c for c in df.columns if "골관절" in c and "약" not in c]

            if htn_cols or dm_cols:
                health_df = df
                break

        # Parse lifestyle
        life_df = None
        for ls in lifestyle_sheets:
            df = pd.read_excel(fpath, sheet_name=ls)
            if "연구대상자ID" in df.columns:
                life_df = df
                break

        if health_df is None:
            continue

        for _, row in health_df.iterrows():
            sid_raw = row.get("연구대상자ID", "")
            if pd.isna(sid_raw):
                continue
            # Normalize ID: 22-MB-0001 -> 22_MB_001
            sid = str(sid_raw).replace("-", "_")
            # Remove leading zeros in the number part
            parts = sid.split("_")
            if len(parts) == 3:
                sid = f"{parts[0]}_{parts[1]}_{int(parts[2]):03d}"

            entry = {"sample_id": sid, "year": year}

            # HTN
            for c in htn_cols:
                val = row.get(c, "")
                if pd.notna(val):
                    entry["hypertension"] = 1 if "예" in str(val) or val == 1.0 else 0

            # DM
            for c in dm_cols:
                val = row.get(c, "")
                if pd.notna(val):
                    entry["diabetes"] = 1 if "예" in str(val) or val == 1.0 else 0

            # OA
            for c in oa_cols:
                val = row.get(c, "")
                if pd.notna(val):
                    entry["osteoarthritis"] = 1 if "예" in str(val) or val == 1.0 else 0

            all_rows.append(entry)

        # Merge lifestyle data (exercise, smoking)
        if life_df is not None:
            exercise_cols = [c for c in life_df.columns if "운동" in c or "스포츠" in c]
            smoke_cols = [c for c in life_df.columns if "담배" in c and "전자" not in c]
            for _, row in life_df.iterrows():
                sid_raw = row.get("연구대상자ID", "")
                if pd.isna(sid_raw):
                    continue
                sid = str(sid_raw).replace("-", "_")
                parts = sid.split("_")
                if len(parts) == 3:
                    sid = f"{parts[0]}_{parts[1]}_{int(parts[2]):03d}"

                # Find matching entry
                for entry in all_rows:
                    if entry["sample_id"] == sid:
                        for c in exercise_cols:
                            val = row.get(c, np.nan)
                            if pd.notna(val) and isinstance(val, (int, float)):
                                entry["exercise_days"] = int(val)
                        for c in smoke_cols:
                            val = row.get(c, "")
                            if pd.notna(val) and "피운 적 없" in str(val):
                                entry["never_smoker"] = 1
                            elif pd.notna(val) and ("5갑" in str(val) or "매일" in str(val)):
                                entry["never_smoker"] = 0
                        break

    if not all_rows:
        return pd.DataFrame()

    result = pd.DataFrame(all_rows).set_index("sample_id")
    # If duplicate sample IDs (from questionnaire vs microbiome mismatch), keep first
    result = result[~result.index.duplicated(keep="first")]
    print(f"  MB questionnaire loaded: {len(result)} subjects "
          f"(HTN: {result['hypertension'].notna().sum()}, "
          f"DM: {result['diabetes'].notna().sum()})")
    return result


def load_all_data():
    """Load and combine DS + Muan data."""
    # Genus abundance for all samples
    genus_rel = load_genus_abundance(LONGEVITY_ASV_TABLE, LONGEVITY_TAXONOMY)

    # DS metadata from KMEDATA
    kme = load_kmedata()
    ds_ids = [x for x in kme.index if isinstance(x, str) and "_DS_" in x and x in genus_rel.index]
    ds_meta = pd.DataFrame({
        "age": pd.to_numeric(kme.loc[ds_ids, "AGE"], errors="coerce"),
        "sex": kme.loc[ds_ids, "SEX"],
        "cohort": "DS",
    })

    # MB metadata (no deduplication; up to ~20 subjects may appear in
    # multiple years but cannot be reliably confirmed without unique IDs)
    mb_raw = load_mb_metadata()
    common_mb = mb_raw.index.intersection(genus_rel.index)
    mb_meta = mb_raw.loc[common_mb]
    # Remove subjects with implausible ages (data entry errors)
    mb_meta = mb_meta[(mb_meta["age"] >= 50) & (mb_meta["age"] <= 110)]
    print(f"  MB with microbiome: {len(mb_meta)}")
    mb_df = pd.DataFrame({
        "age": mb_meta["age"],
        "sex": mb_meta["sex"],
        "cohort": "Muan",
    })

    # Combine
    meta = pd.concat([ds_meta, mb_df])
    meta = meta.dropna(subset=["age"])

    # Filter genus table to combined samples
    common = meta.index.intersection(genus_rel.index)
    meta = meta.loc[common]
    genus_rel = genus_rel.loc[common]

    # Sex harmonization (DS uses M/F, MB uses M/F already)
    print(f"  DS: n={sum(meta.cohort=='DS')}, age {meta.loc[meta.cohort=='DS','age'].median():.0f} "
          f"({meta.loc[meta.cohort=='DS','age'].min():.0f}-{meta.loc[meta.cohort=='DS','age'].max():.0f})")
    print(f"  Muan: n={sum(meta.cohort=='Muan')}, age {meta.loc[meta.cohort=='Muan','age'].median():.0f} "
          f"({meta.loc[meta.cohort=='Muan','age'].min():.0f}-{meta.loc[meta.cohort=='Muan','age'].max():.0f})")

    # Load MB questionnaire data
    mb_quest = load_mb_questionnaire()

    return meta, genus_rel, mb_quest


def get_age_groups(meta):
    """Assign subjects to analysis groups."""
    groups = pd.Series("", index=meta.index)
    groups[(meta["cohort"] == "Muan")] = "RC elderly"
    groups[(meta["cohort"] == "DS") & (meta["age"] >= 60)] = "DS elderly"
    groups[(meta["cohort"] == "DS") & (meta["age"] < 40)] = "DS young"
    groups[(meta["cohort"] == "DS") & (meta["age"] >= 40) & (meta["age"] < 60)] = "DS middle"
    return groups


# ===================================================================
# FIG 1: COHORT OVERVIEW
# ===================================================================

def fig1_cohort_overview(meta, mb_quest):
    print("  Fig 1: Cohort overview...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Age distributions
    ax = axes[0]
    ds = meta[meta["cohort"] == "DS"]
    muan = meta[meta["cohort"] == "Muan"]
    bins = np.arange(20, 105, 5)
    ax.hist(ds["age"], bins=bins, alpha=0.6, label=f"DS (n={len(ds)})",
            color=COLORS["ds_old"], density=True)
    ax.hist(muan["age"], bins=bins, alpha=0.6, label=f"RC (n={len(muan)})",
            color=COLORS["muan"], density=True)
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Density")
    ax.set_title("A. Age Distribution by Cohort")
    ax.legend()

    # Panel B: Sex distribution by cohort
    ax = axes[1]
    cohort_display = meta["cohort"].replace({"Muan": "RC"})
    ct = pd.crosstab(cohort_display, meta["sex"])
    ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
    ct_pct.plot(kind="bar", stacked=True, ax=ax,
                color=["#2196F3", "#E53935"], alpha=0.8)
    ax.set_ylabel("Percentage")
    ax.set_title("B. Sex Distribution by Cohort")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    cohort_n = {"DS": (meta["cohort"] == "DS").sum(), "RC": (meta["cohort"] == "Muan").sum()}
    for i, (idx, row) in enumerate(ct.iterrows()):
        n_display = cohort_n.get(idx, int(row.sum()))
        ax.text(i, 50, f"n={n_display}", ha="center", va="center",
                fontweight="bold", color="white", fontsize=11)
    ax.legend(title="Sex")


    plt.tight_layout()
    fig.savefig(OUT / "fig1_cohort_overview.png")
    fig.savefig(OUT / "fig1_cohort_overview.pdf")
    plt.close()

    # Summary table with disease prevalence for Muan
    muan_ids = meta[meta["cohort"] == "Muan"].index
    muan_quest = mb_quest.reindex(muan_ids) if len(mb_quest) > 0 else pd.DataFrame()

    summary = []
    for cohort in ["DS", "Muan"]:
        c = meta[meta["cohort"] == cohort]
        row = {
            "cohort": cohort, "n": len(c),
            "age_mean": c["age"].mean(), "age_std": c["age"].std(),
            "age_min": c["age"].min(), "age_max": c["age"].max(),
            "pct_female": (c["sex"] == "F").mean() * 100,
        }
        if cohort == "Muan" and len(muan_quest) > 0:
            for disease in ["hypertension", "diabetes", "osteoarthritis"]:
                if disease in muan_quest.columns:
                    vals = muan_quest[disease].dropna()
                    row[f"{disease}_n"] = int(vals.sum())
                    row[f"{disease}_pct"] = vals.mean() * 100 if len(vals) > 0 else np.nan
                    row[f"{disease}_total"] = len(vals)
            if "exercise_days" in muan_quest.columns:
                ex = muan_quest["exercise_days"].dropna()
                row["exercise_days_mean"] = ex.mean()
            if "never_smoker" in muan_quest.columns:
                ns = muan_quest["never_smoker"].dropna()
                row["never_smoker_pct"] = ns.mean() * 100
        summary.append(row)
    pd.DataFrame(summary).to_csv(OUT / "table1_cohort_summary.csv", index=False)


# ===================================================================
# FIG 2: BETA DIVERSITY (PCoA + PERMANOVA)
# ===================================================================

def _bray_curtis(rel_df):
    """Compute Bray-Curtis distance matrix from relative abundance."""
    return squareform(pdist(rel_df.values, metric="braycurtis"))


def _permanova(dist_matrix, groups, n_perm=999):
    """Simple PERMANOVA (one-way) on precomputed distance matrix."""
    labels = np.array(groups)
    unique_groups = np.unique(labels)
    n = len(labels)

    # Squared distances
    D2 = dist_matrix ** 2

    def pseudo_f(lab):
        # Total sum of squares
        ss_total = D2.sum() / (2 * n)
        # Within-group SS
        ss_within = 0
        for g in unique_groups:
            mask = lab == g
            ng = mask.sum()
            if ng > 1:
                ss_within += D2[np.ix_(mask, mask)].sum() / (2 * ng)
        ss_between = ss_total - ss_within
        k = len(unique_groups)
        f_stat = (ss_between / (k - 1)) / (ss_within / (n - k))
        return f_stat

    f_obs = pseudo_f(labels)
    n_ge = 1  # include observed
    for _ in range(n_perm):
        perm_labels = np.random.permutation(labels)
        if pseudo_f(perm_labels) >= f_obs:
            n_ge += 1
    p_value = n_ge / (n_perm + 1)
    return f_obs, p_value


def fig2_beta_diversity(meta, genus_rel):
    print("  Fig 2: Beta diversity (PCoA + PERMANOVA)...")
    # Use Bray-Curtis on relative abundance
    bc_dist = _bray_curtis(genus_rel)

    # PCoA via classical MDS (eigendecomposition of centered distance matrix)
    n = bc_dist.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * H @ (bc_dist ** 2) @ H
    eigvals, eigvecs = np.linalg.eigh(B)
    # Take top 2 positive eigenvalues
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    # Keep only positive eigenvalues
    pos = eigvals > 0
    pc1 = eigvecs[:, 0] * np.sqrt(eigvals[0])
    pc2 = eigvecs[:, 1] * np.sqrt(eigvals[1])
    var_exp = eigvals[:2] / eigvals[pos].sum() * 100

    pcoa_df = pd.DataFrame({"PC1": pc1, "PC2": pc2}, index=genus_rel.index)
    pcoa_df["cohort"] = meta["cohort"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: PCoA colored by cohort
    ax = axes[0]
    for cohort, label, color in [("DS", "DS", COLORS["ds_light"]), ("Muan", "RC", COLORS["muan_light"])]:
        mask = pcoa_df["cohort"] == cohort
        ax.scatter(pcoa_df.loc[mask, "PC1"], pcoa_df.loc[mask, "PC2"],
                   c=color, alpha=0.4, s=8, label=label)
    ax.set_xlabel(f"PCoA 1 ({var_exp[0]:.1f}%)")
    ax.set_ylabel(f"PCoA 2 ({var_exp[1]:.1f}%)")
    n_total = len(pcoa_df)
    ax.set_title(f"A. PCoA (Bray-Curtis) by Cohort (n = {n_total})")
    ax.legend(markerscale=3)

    # Panel B: PCoA colored by age
    ax = axes[1]
    ages = meta.loc[pcoa_df.index, "age"]
    sc = ax.scatter(pcoa_df["PC1"], pcoa_df["PC2"], c=ages, cmap="viridis",
                    alpha=0.4, s=8, vmin=20, vmax=100)
    plt.colorbar(sc, ax=ax, label="Age (years)")
    ax.set_xlabel(f"PCoA 1 ({var_exp[0]:.1f}%)")
    ax.set_ylabel(f"PCoA 2 ({var_exp[1]:.1f}%)")
    ax.set_title(f"B. PCoA (Bray-Curtis) by Age (n = {n_total})")

    # PERMANOVA: cohort effect
    f_stat, p_val = _permanova(bc_dist, meta["cohort"].values)
    # Compute R² from distance matrix directly
    D2 = bc_dist ** 2
    n_all = len(meta)
    ss_total = D2.sum() / (2 * n_all)
    labels = meta["cohort"].values
    ss_within = 0
    for g in np.unique(labels):
        mask = labels == g
        ng = mask.sum()
        ss_within += D2[np.ix_(mask, mask)].sum() / (2 * ng)
    r2_cohort = (ss_total - ss_within) / ss_total * 100
    axes[0].text(0.02, 0.98, f"PERMANOVA\nF={f_stat:.1f}, R²={r2_cohort:.2f}%, P={p_val:.3f}",
                 transform=axes[0].transAxes, va="top", fontsize=8,
                 bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    # PERMANOVA: age group effect (Muan vs DS elderly)
    groups = get_age_groups(meta)
    age_matched = groups.isin(["RC elderly", "DS elderly"])
    if age_matched.sum() > 20:
        bc_sub = _bray_curtis(genus_rel.loc[age_matched])
        f2, p2 = _permanova(bc_sub, groups[age_matched].values)
        n_sub = age_matched.sum()
        D2_sub = bc_sub ** 2
        ss_t2 = D2_sub.sum() / (2 * n_sub)
        lab2 = groups[age_matched].values
        ss_w2 = 0
        for g in np.unique(lab2):
            mask = lab2 == g
            ng = mask.sum()
            ss_w2 += D2_sub[np.ix_(mask, mask)].sum() / (2 * ng)
        r2_sub = (ss_t2 - ss_w2) / ss_t2 * 100
        axes[0].text(0.02, 0.78,
                     f"RC vs DS older adults\nF={f2:.1f}, R²={r2_sub:.2f}%, P={p2:.3f}",
                     transform=axes[0].transAxes, va="top", fontsize=8,
                     bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))


    plt.tight_layout()
    fig.savefig(OUT / "fig2_beta_diversity.png")
    fig.savefig(OUT / "fig2_beta_diversity.pdf")
    plt.close()

    return {"permanova_cohort_F": f_stat, "permanova_cohort_P": p_val}


# ===================================================================
# FIG 3: ALPHA DIVERSITY
# ===================================================================

def fig3_alpha_diversity(meta, genus_rel):
    print("  Fig 3: Alpha diversity...")
    groups = get_age_groups(meta)
    keep = groups.isin(["DS young", "DS elderly", "RC elderly"])
    meta_k = meta.loc[keep]
    genus_k = genus_rel.loc[keep]
    groups_k = groups[keep]

    # Shannon diversity
    shannon = genus_k.apply(lambda row: stats.entropy(row[row > 0]), axis=1)
    observed = (genus_k > 0).sum(axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    group_order = ["DS young", "DS elderly", "RC elderly"]
    group_colors = [COLORS["ds_young"], COLORS["ds_old"], COLORS["muan"]]

    for ax, metric, label in [(axes[0], shannon, "Shannon Diversity"),
                               (axes[1], observed, "Observed Genera")]:
        data = [metric[groups_k == g].values for g in group_order]
        # n and median go in the tick labels so they cannot collide with outliers
        display_labels = [f"{name}\n(n={len(d):,}; median {np.median(d):.{0 if np.median(d) % 1 == 0 else 2}f})"
                          for name, d in zip(["DS young", "DS older adults", "RC older adults"], data)]
        bp = ax.boxplot(data, labels=display_labels, patch_artist=True, widths=0.6)
        for patch, color in zip(bp["boxes"], group_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.set_ylabel(label)
        ax.tick_params(axis="x", labelsize=9)
        # Headroom so the pairwise P (upper right) clears the top whisker/outliers
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax + 0.10 * (ymax - ymin))

        # Kruskal-Wallis
        h_stat, h_p = stats.kruskal(*data)
        ax.set_title(f"{label}\n(Kruskal-Wallis P={h_p:.4f})")

        # Pairwise Mann-Whitney between DS elderly and RC elderly
        u, p_mw = stats.mannwhitneyu(data[1], data[2], alternative="two-sided")
        ax.text(0.98, 0.98, f"DS older adults vs RC: P={p_mw:.4f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=8)


    plt.tight_layout()
    fig.savefig(OUT / "fig3_alpha_diversity.png")
    fig.savefig(OUT / "fig3_alpha_diversity.pdf")
    plt.close()


# ===================================================================
# FIG 4: DIFFERENTIAL ABUNDANCE (Muan vs DS elderly)
# ===================================================================

def fig4_differential_abundance(meta, genus_rel):
    print("  Fig 4: Differential abundance (Muan vs DS elderly)...")
    groups = get_age_groups(meta)
    muan = groups[groups == "RC elderly"].index
    ds_old = groups[groups == "DS elderly"].index

    genus_clr = clr_transform(genus_rel)
    # Filter to prevalent genera (>10% prevalence overall)
    prevalent = (genus_rel > 0).mean(axis=0)
    prev_genera = prevalent[prevalent >= 0.10].index.tolist()

    results = []
    for genus in prev_genera:
        m_vals = genus_clr.loc[muan, genus].values
        d_vals = genus_clr.loc[ds_old, genus].values
        stat, p = stats.mannwhitneyu(m_vals, d_vals, alternative="two-sided")
        # Cliff's delta
        n_m, n_d = len(m_vals), len(d_vals)
        delta = (np.sum(m_vals[:, None] > d_vals[None, :])
                 - np.sum(m_vals[:, None] < d_vals[None, :])) / (n_m * n_d)
        results.append({
            "genus": genus,
            "clr_diff": m_vals.mean() - d_vals.mean(),
            "mean_muan": m_vals.mean(),
            "mean_ds_elderly": d_vals.mean(),
            "p_value": p,
            "cliffs_delta": delta,
        })

    df = pd.DataFrame(results)
    _, df["q_value"], _, _ = multipletests(df["p_value"], method="fdr_bh")
    df = df.sort_values("p_value")
    df.to_csv(OUT / "table3_muan_vs_ds_elderly.csv", index=False)

    n_sig = (df["q_value"] < 0.05).sum()
    print(f"    {n_sig} genera differ (FDR < 0.05)")
    print(f"    LC elderly: n={len(muan)}, DS elderly: n={len(ds_old)}")

    # Plot top 25: two-panel figure (CLR diff + Cliff's delta)
    top = df.head(25).sort_values("clr_diff")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8),
                                    gridspec_kw={"width_ratios": [3, 2]})

    # Panel A: CLR differences
    colors = [COLORS["muan"] if d > 0 else COLORS["ds_old"] for d in top["clr_diff"]]
    ax1.barh(range(len(top)), top["clr_diff"], color=colors, alpha=0.8)
    ax1.set_yticks(range(len(top)))
    ylabels = [f"{r['genus']} *" if r["q_value"] < 0.05 else r["genus"]
               for _, r in top.iterrows()]
    ax1.set_yticklabels(ylabels, fontsize=8)
    ax1.axvline(0, color="black", linewidth=0.5)
    ax1.set_xlabel("CLR Difference (RC − DS older adults)")
    ax1.set_title("A. CLR Difference")

    # Panel B: Cliff's delta
    delta_colors = [COLORS["muan"] if d > 0 else COLORS["ds_old"]
                    for d in top["cliffs_delta"]]
    ax2.barh(range(len(top)), top["cliffs_delta"], color=delta_colors, alpha=0.8)
    ax2.set_yticks(range(len(top)))
    ax2.set_yticklabels([])
    ax2.axvline(0, color="black", linewidth=0.5)
    # Negligible threshold lines
    ax2.axvline(-0.147, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
    ax2.axvline(0.147, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
    ax2.set_xlabel("Cliff's δ (effect size)")
    ax2.set_title("B. Cliff's δ")

    fig.suptitle(f"Top 25 Differentially Abundant Genera\n"
                 f"RC (n={len(muan)}) vs DS Older Adults (n={len(ds_old)}), "
                 f"{n_sig} significant (FDR < 0.05)", fontsize=11)
    # Legend
    from matplotlib.patches import Patch
    ax1.legend(handles=[Patch(color=COLORS["muan"], label="Higher in RC"),
                        Patch(color=COLORS["ds_old"], label="Higher in DS")],
               loc="lower right", fontsize=8)
    ax2.legend(handles=[Patch(facecolor="none", edgecolor="gray", linestyle="--",
                              label="|δ| = 0.147 (negligible)")],
               loc="lower right", fontsize=7)
    plt.tight_layout()
    fig.savefig(OUT / "fig2_differential_abundance.png", dpi=300)
    fig.savefig(OUT / "fig2_differential_abundance.pdf")
    plt.close()
    return df


# ===================================================================
# FIG 5: ENTEROTYPE ANALYSIS
# ===================================================================

def _pam(dist_matrix, k, max_iter=100, random_state=42):
    """Partitioning Around Medoids on a precomputed distance matrix."""
    rng = np.random.RandomState(random_state)
    n = dist_matrix.shape[0]
    # Initialize medoids randomly
    medoids = rng.choice(n, k, replace=False)
    for _ in range(max_iter):
        # Assign each point to nearest medoid
        labels = np.argmin(dist_matrix[:, medoids], axis=1)
        # Update medoids
        new_medoids = np.zeros(k, dtype=int)
        for c in range(k):
            members = np.where(labels == c)[0]
            if len(members) == 0:
                new_medoids[c] = medoids[c]
                continue
            costs = dist_matrix[np.ix_(members, members)].sum(axis=1)
            new_medoids[c] = members[np.argmin(costs)]
        if np.array_equal(sorted(new_medoids), sorted(medoids)):
            break
        medoids = new_medoids
    labels = np.argmin(dist_matrix[:, medoids], axis=1)
    return medoids, labels


def _assign_enterotypes(genus_rel, n_clusters=2):
    """Enterotype assignment using PAM on Jensen-Shannon divergence,
    following Arumugam et al. (2011)."""
    rel = genus_rel.copy()
    rel = rel + 1e-6
    rel = rel.div(rel.sum(axis=1), axis=0)

    # Compute JSD distance matrix (vectorized for speed)
    from scipy.spatial.distance import pdist, squareform
    vals = rel.values
    log_vals = np.log(vals)
    n = len(vals)
    print(f"    Computing JSD for {n} samples...")
    jsd_condensed = np.zeros(n * (n - 1) // 2)
    idx = 0
    for i in range(n):
        if i % 200 == 0 and i > 0:
            print(f"      {i}/{n}...")
        for j in range(i + 1, n):
            m = 0.5 * (vals[i] + vals[j])
            log_m = np.log(m)
            kl_pm = np.sum(vals[i] * (log_vals[i] - log_m))
            kl_qm = np.sum(vals[j] * (log_vals[j] - log_m))
            jsd_condensed[idx] = np.sqrt(0.5 * (kl_pm + kl_qm))
            idx += 1
    jsd_matrix = squareform(jsd_condensed)

    # PAM clustering, select best K by silhouette
    from sklearn.metrics import silhouette_score
    best_k = 2
    best_sil = -1
    best_labels = None
    for k in range(2, 5):
        _, labels = _pam(jsd_matrix, k, random_state=RANDOM_STATE)
        sil = silhouette_score(jsd_matrix, labels, metric="precomputed")
        print(f"    PAM K={k}: silhouette={sil:.4f}")
        if sil > best_sil:
            best_sil = sil
            best_k = k
            best_labels = labels
    print(f"    Best K={best_k} (silhouette={best_sil:.4f})")

    # Label enterotypes by dominant genus
    et_labels = {}
    for e in range(best_k):
        members = rel.index[best_labels == e]
        dominant = genus_rel.loc[members].mean().idxmax()
        clean = re.sub(r"_\d+$", "", dominant)
        et_labels[e] = clean

    et = pd.Series(best_labels, index=genus_rel.index, name="enterotype")
    return et, et_labels


def fig5_enterotypes(meta, genus_rel):
    print("  Fig 5: Enterotype analysis...")
    et, et_labels = _assign_enterotypes(genus_rel)

    # Rename to ET1/ET2 with dominant genus
    et_named = et.map(lambda x: f"ET{x+1}: {et_labels[x]}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Enterotype proportion by group
    ax = axes[0]
    groups = get_age_groups(meta)
    keep = groups.isin(["DS young", "DS elderly", "RC elderly"])
    ct = pd.crosstab(groups[keep], et_named[keep], normalize="index") * 100
    group_order = ["DS young", "DS elderly", "RC elderly"]
    ct = ct.reindex(group_order)
    ct.plot(kind="bar", stacked=True, ax=ax, alpha=0.85,
            color=["#2196F3", "#E53935", "#4CAF50", "#FF9800"][:len(ct.columns)])
    ax.set_ylabel("Percentage (%)")
    ax.set_xlabel("")
    ax.set_title("A. Enterotype Distribution by Group")
    ax.set_xticklabels(group_order, rotation=0, fontsize=9)
    ax.legend(title="Enterotype", fontsize=8)

    # Chi-square test
    ct_raw = pd.crosstab(groups[keep], et[keep])
    ct_raw = ct_raw.reindex(group_order)
    chi2, p_chi, _, _ = stats.chi2_contingency(ct_raw)
    ax.text(0.98, 0.02, f"Chi-square P={p_chi:.4f}", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=9)

    # Panel B: Enterotype proportion by cohort and age decade
    ax = axes[1]
    decade = (meta["age"] // 10 * 10).astype(int)
    cohort_decade = meta["cohort"] + "_" + decade.astype(str)
    ct2 = pd.crosstab(cohort_decade, et_named, normalize="index") * 100

    # Plot only decades with enough data
    decade_order = []
    for d in range(20, 100, 10):
        for c in ["DS", "Muan"]:  # data values
            key = f"{c}_{d}"
            if key in ct2.index and pd.crosstab(cohort_decade, et).loc[key].sum() >= 10:
                decade_order.append(key)

    if decade_order:
        ct2 = ct2.reindex(decade_order)
        ct2.plot(kind="bar", stacked=True, ax=ax, alpha=0.85,
                 color=["#2196F3", "#E53935", "#4CAF50", "#FF9800"][:len(ct2.columns)])
        ax.set_ylabel("Percentage (%)")
        ax.set_xlabel("")
        ax.set_title("B. Enterotype by Cohort and Age Decade")
        ax.set_xticklabels([x.replace("Muan", "RC").replace("_", " ") for x in decade_order],
                           rotation=45, ha="right", fontsize=8)
        ax.legend(title="Enterotype", fontsize=8)


    plt.tight_layout()
    fig.savefig(OUT / "fig5_enterotypes.png")
    fig.savefig(OUT / "fig5_enterotypes.pdf")
    plt.close()

    # Table
    ct_save = pd.crosstab(groups[keep], et_named)
    ct_save.to_csv(OUT / "table4_enterotypes.csv")
    return et, et_labels


# ===================================================================
# FIG 6: MICROBIOME AGE (train on DS, predict Muan)
# ===================================================================

def fig6_microbiome_age(meta, genus_rel):
    print("  Fig 6: Microbiome age prediction...")
    genus_top = select_top_genera(genus_rel, 50)
    X_all = clr_transform(genus_top)

    # Split into DS (training) and Muan (test)
    ds_idx = meta[meta["cohort"] == "DS"].index.intersection(X_all.index)
    muan_idx = meta[meta["cohort"] == "Muan"].index.intersection(X_all.index)

    X_ds = X_all.loc[ds_idx]
    y_ds = meta.loc[ds_idx, "age"]
    X_muan = X_all.loc[muan_idx]
    y_muan = meta.loc[muan_idx, "age"]

    # 5-fold CV on DS to get DS predictions + train final model
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    ds_pred = np.zeros(len(y_ds))

    for train_idx, test_idx in kf.split(X_ds):
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("reg", GradientBoostingRegressor(
                n_estimators=200, max_depth=3, learning_rate=0.05,
                random_state=RANDOM_STATE, subsample=0.8)),
        ])
        pipe.fit(X_ds.iloc[train_idx], y_ds.iloc[train_idx])
        ds_pred[test_idx] = pipe.predict(X_ds.iloc[test_idx])

    # Train final model on all DS, predict Muan
    final_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("reg", GradientBoostingRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            random_state=RANDOM_STATE, subsample=0.8)),
    ])
    final_pipe.fit(X_ds, y_ds)
    muan_pred = final_pipe.predict(X_muan)

    ds_mae = mean_absolute_error(y_ds, ds_pred)
    muan_mae = mean_absolute_error(y_muan, muan_pred)
    ds_delta = ds_pred - y_ds.values
    muan_delta = muan_pred - y_muan.values

    print(f"    DS MAE: {ds_mae:.1f}y, Muan MAE: {muan_mae:.1f}y")
    print(f"    DS delta mean: {ds_delta.mean():+.1f}, Muan delta mean: {muan_delta.mean():+.1f}")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel A: Predicted vs actual (DS)
    ax = axes[0]
    ax.scatter(y_ds, ds_pred, alpha=0.3, s=10, color=COLORS["ds_old"])
    ax.plot([20, 100], [20, 100], "k--", linewidth=1)
    rho_ds, _ = stats.spearmanr(y_ds, ds_pred)
    ax.set_xlabel("Chronological Age")
    ax.set_ylabel("Predicted Microbiome Age")
    ax.set_title("A. DS (5-fold CV)\n" + f"MAE={ds_mae:.1f}y, " + r"$\rho$" + f"={rho_ds:.2f}")

    # Panel B: Predicted vs actual (Muan)
    ax = axes[1]
    ax.scatter(y_muan, muan_pred, alpha=0.3, s=10, color=COLORS["muan"])
    ax.plot([20, 100], [20, 100], "k--", linewidth=1)
    rho_muan, _ = stats.spearmanr(y_muan, muan_pred)
    ax.set_xlabel("Chronological Age")
    ax.set_ylabel("Predicted Microbiome Age")
    ax.set_title("B. RC (external prediction)\n" + f"MAE={muan_mae:.1f}y, " + r"$\rho$" + f"={rho_muan:.2f}")

    # Panel C: Delta age distribution comparison
    ax = axes[2]
    # Compare delta for age-matched groups only
    groups = get_age_groups(meta)
    ds_elderly_idx = groups[groups == "DS elderly"].index.intersection(pd.Index(y_ds.index))
    muan_elderly_idx = groups[groups == "RC elderly"].index.intersection(pd.Index(y_muan.index))

    ds_elderly_delta = ds_pred[np.isin(y_ds.index, ds_elderly_idx)] - y_ds.loc[ds_elderly_idx].values
    muan_elderly_delta = muan_pred[np.isin(y_muan.index, muan_elderly_idx)] - y_muan.loc[muan_elderly_idx].values

    ax.hist(ds_elderly_delta, bins=20, alpha=0.6, color=COLORS["ds_old"],
            label=f"DS elderly (mean={ds_elderly_delta.mean():+.1f})", density=True)
    ax.hist(muan_elderly_delta, bins=20, alpha=0.6, color=COLORS["muan"],
            label=f"RC elderly (mean={muan_elderly_delta.mean():+.1f})", density=True)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("Microbiome Age Gap (predicted − actual)")
    ax.set_ylabel("Density")
    u_stat, p_delta = stats.mannwhitneyu(ds_elderly_delta, muan_elderly_delta, alternative="two-sided")
    ax.set_title(f"C. Age Gap: DS vs RC Elderly\n(Mann-Whitney P={p_delta:.4f})")
    ax.legend(fontsize=8)


    plt.tight_layout()
    fig.savefig(OUT / "fig6_microbiome_age.png")
    fig.savefig(OUT / "fig6_microbiome_age.pdf")
    plt.close()

    # Save results
    results = pd.concat([
        pd.DataFrame({"sample_id": y_ds.index, "cohort": "DS",
                       "chronological_age": y_ds.values, "predicted_age": ds_pred,
                       "delta_age": ds_delta}),
        pd.DataFrame({"sample_id": y_muan.index, "cohort": "Muan",
                       "chronological_age": y_muan.values, "predicted_age": muan_pred,
                       "delta_age": muan_delta}),
    ])
    results.to_csv(OUT / "table5_microbiome_age.csv", index=False)

    return {
        "ds_mae": ds_mae, "muan_mae": muan_mae,
        "ds_delta_mean": ds_delta.mean(), "muan_delta_mean": muan_delta.mean(),
        "delta_p": p_delta,
    }


# ===================================================================
# FIG 7: KEY GENERA COMPARISON (boxplots)
# ===================================================================

def fig7_key_genera(meta, genus_rel, da_results):
    print("  Fig 7: Key genera boxplots...")
    groups = get_age_groups(meta)
    keep_mask = groups.isin(["DS young", "DS elderly", "RC elderly"])
    keep_idx = groups[keep_mask].index
    genus_clr = clr_transform(genus_rel)
    groups_k = groups.loc[keep_idx]

    # Curated 14 genera: top effects + DS-60+ outliers + key Discussion genera
    # Ensures all genera referenced in the text as "see Figure 3" are present
    must_include = [
        # Core lactobacilli enrichment (true cohort effects)
        "Limosilactobacillus", "Lactobacillus", "Megasphaera",
        # DS-60+ outlier pattern (text sends reader here)
        "Roseburia", "Coprococcus", "[Eubacterium] eligens group", "Haemophilus",
        # Key depletions (true cohort effects)
        "Lachnoclostridium", "Lachnospiraceae UCG-004",
        # Other enriched discussed in text
        "Escherichia-Shigella", "Olsenella", "Methanobrevibacter",
        # Large CLR differences
        "Erysipelotrichaceae UCG-003", "Intestinibacter",
    ]
    genera = [g for g in must_include if g in genus_clr.columns]

    n_genera = len(genera)
    ncols = 4
    nrows = (n_genera + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4 * nrows))
    axes = axes.flatten()

    group_order = ["DS young", "DS elderly", "RC elderly"]
    group_colors = [COLORS["ds_young"], COLORS["ds_old"], COLORS["muan"]]

    for i, genus in enumerate(genera):
        ax = axes[i]
        data = [genus_clr.loc[groups_k[groups_k == g].index, genus].values for g in group_order]
        bp = ax.boxplot(data, labels=[f"DS young\n(n={len(data[0])})",
                                       f"DS older adults\n(n={len(data[1])})",
                                       f"RC older adults\n(n={len(data[2])})"],
                        patch_artist=True, widths=0.6)
        for patch, color in zip(bp["boxes"], group_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        # Add group means as diamonds
        means = [d.mean() for d in data]
        ax.scatter([1, 2, 3], means, marker="D", color="black", s=20, zorder=5)

        # Get q-value from DA results (RC vs DS elderly)
        q_val = None
        if da_results is not None:
            row = da_results[da_results["genus"] == genus]
            if len(row) > 0:
                q_val = row.iloc[0]["q_value"]

        # Clean genus name for title
        clean = re.sub(r"_\d+$", "", genus)
        q_str = f"\nq = {q_val:.2e}" if q_val is not None else ""
        ax.set_title(f"{clean}{q_str}", fontsize=8)
        ax.set_ylabel("CLR Abundance", fontsize=7)
        ax.tick_params(axis="x", labelsize=7)

    # Hide unused axes
    for i in range(n_genera, len(axes)):
        axes[i].set_visible(False)


    plt.tight_layout()
    fig.savefig(OUT / "fig7_key_genera.png")
    fig.savefig(OUT / "fig7_key_genera.pdf")
    plt.close()


# ===================================================================
# TABLE: PERMANOVA (cohort, age, sex)
# ===================================================================

def table_permanova(meta, genus_rel):
    print("  Table: Multi-factor PERMANOVA...")
    bc_dist = _bray_curtis(genus_rel)

    results = []
    for factor in ["cohort", "sex"]:
        valid = meta[factor].dropna().index
        bc_sub = _bray_curtis(genus_rel.loc[valid])
        f_stat, p_val = _permanova(bc_sub, meta.loc[valid, factor].values)
        results.append({"factor": factor, "F": f_stat, "P": p_val, "n": len(valid)})

    # Age groups
    groups = get_age_groups(meta)
    keep = groups != ""
    bc_sub = _bray_curtis(genus_rel.loc[keep])
    f_stat, p_val = _permanova(bc_sub, groups[keep].values)
    results.append({"factor": "age_group", "F": f_stat, "P": p_val, "n": keep.sum()})

    pd.DataFrame(results).to_csv(OUT / "table2_permanova.csv", index=False)
    for r in results:
        print(f"    {r['factor']}: F={r['F']:.2f}, P={r['P']:.3f}")


# ===================================================================
# MAIN
# ===================================================================

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("Longevity Village Analysis: Muan-gun vs DS")
    print("=" * 50)

    meta, genus_rel, mb_quest = load_all_data()
    log = {"n_ds": int(sum(meta.cohort == "DS")),
           "n_muan": int(sum(meta.cohort == "Muan"))}

    fig1_cohort_overview(meta, mb_quest)
    perm = fig2_beta_diversity(meta, genus_rel)
    log.update(perm)

    fig3_alpha_diversity(meta, genus_rel)
    da = fig4_differential_abundance(meta, genus_rel)
    log["n_da_sig"] = int((da["q_value"] < 0.05).sum())

    et, et_labels = fig5_enterotypes(meta, genus_rel)
    age_results = fig6_microbiome_age(meta, genus_rel)
    log.update(age_results)

    fig7_key_genera(meta, genus_rel, da)
    table_permanova(meta, genus_rel)

    with open(OUT / "summary_log.json", "w") as f:
        json.dump(log, f, indent=2, default=str)

    print(f"\nAll saved to {OUT}")
    print(f"Summary: {json.dumps(log, indent=2, default=str)}")


if __name__ == "__main__":
    main()

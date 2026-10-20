"""
Additional analyses for the longevity village manuscript.

1. Feature importances: which genera drive the "young" age prediction
2. Age-genus overlap: DS age-associated genera behavior in Muan
3. Diet-microbiome link: fermented food frequency vs Lactobacillus (25_MB)
4. Exercise/smoking vs microbiome age gap

Usage (set paths in ml/longevity_config.py first):
    cd /path/to/kme-microbiome-analysis
    python -m ml.longevity_extra
"""
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

from ml.config import RANDOM_STATE
from ml.data_join import clr_transform, load_genus_abundance, load_kmedata, select_top_genera
from ml.longevity_config import LONGEVITY_ASV_TABLE, LONGEVITY_OUT, LONGEVITY_TAXONOMY
from ml.longevity_muan import load_mb_metadata, load_mb_questionnaire, get_age_groups, COLORS

warnings.filterwarnings("ignore")

OUT = LONGEVITY_OUT

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 11,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.family": "sans-serif",
})


def load_data():
    """Load combined DS + Muan data (same as longevity_muan but simpler)."""
    genus_rel = load_genus_abundance(LONGEVITY_ASV_TABLE, LONGEVITY_TAXONOMY)
    kme = load_kmedata()
    ds_ids = [x for x in kme.index if isinstance(x, str) and "_DS_" in x and x in genus_rel.index]
    ds_meta = pd.DataFrame({
        "age": pd.to_numeric(kme.loc[ds_ids, "AGE"], errors="coerce"),
        "sex": kme.loc[ds_ids, "SEX"],
        "cohort": "DS",
    })

    mb_raw = load_mb_metadata()
    common_mb = mb_raw.index.intersection(genus_rel.index)
    mb_meta = mb_raw.loc[common_mb]
    mb_df = pd.DataFrame({
        "age": mb_meta["age"],
        "sex": mb_meta["sex"],
        "cohort": "Muan",
    })

    meta = pd.concat([ds_meta, mb_df]).dropna(subset=["age"])
    common = meta.index.intersection(genus_rel.index)
    meta = meta.loc[common]
    genus_rel = genus_rel.loc[common]
    return meta, genus_rel


# ===================================================================
# 1. FEATURE IMPORTANCES: which genera make Muan look young
# ===================================================================

def fig_feature_importance(meta, genus_rel):
    print("  Analysis 1: Feature importances from age model...")
    genus_top = select_top_genera(genus_rel, 50)
    X_all = clr_transform(genus_top)

    ds_idx = meta[meta["cohort"] == "DS"].index.intersection(X_all.index)
    X_ds = X_all.loc[ds_idx]
    y_ds = meta.loc[ds_idx, "age"]

    # Train final model on all DS
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("reg", GradientBoostingRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            random_state=RANDOM_STATE, subsample=0.8)),
    ])
    pipe.fit(X_ds, y_ds)

    # Extract feature importances
    importances = pipe.named_steps["reg"].feature_importances_
    feat_imp = pd.DataFrame({
        "genus": X_all.columns,
        "importance": importances,
    }).sort_values("importance", ascending=False)

    # For each genus, compute direction: does higher abundance predict older or younger?
    # Use the mean CLR difference between Muan and DS elderly
    groups = get_age_groups(meta)
    muan_idx = groups[groups == "RC elderly"].index.intersection(X_all.index)
    ds_old_idx = groups[groups == "DS elderly"].index.intersection(X_all.index)

    clr_diff = X_all.loc[muan_idx].mean() - X_all.loc[ds_old_idx].mean()
    feat_imp["muan_vs_ds_elderly"] = feat_imp["genus"].map(clr_diff)

    # Compute age correlation in DS
    age_corr = {}
    for genus in X_all.columns:
        rho, _ = stats.spearmanr(y_ds, X_ds[genus])
        age_corr[genus] = rho
    feat_imp["ds_age_rho"] = feat_imp["genus"].map(age_corr)

    # Direction of contribution to "young" prediction:
    # If genus increases with age in DS (positive rho) AND is lower in Muan -> makes Muan look young
    # If genus decreases with age in DS (negative rho) AND is higher in Muan -> makes Muan look young
    feat_imp["young_direction"] = ""
    mask_young = (
        ((feat_imp["ds_age_rho"] > 0) & (feat_imp["muan_vs_ds_elderly"] < 0)) |
        ((feat_imp["ds_age_rho"] < 0) & (feat_imp["muan_vs_ds_elderly"] > 0))
    )
    feat_imp.loc[mask_young, "young_direction"] = "younger"
    feat_imp.loc[~mask_young, "young_direction"] = "older"

    feat_imp.to_csv(OUT / "table_feature_importance.csv", index=False)

    # Plot top 20
    top = feat_imp.head(20).copy()
    top = top.sort_values("importance")

    fig, ax = plt.subplots(figsize=(9, 7))
    colors = [COLORS["ds_young"] if d == "younger" else COLORS["muan"]
              for d in top["young_direction"]]
    ax.barh(range(len(top)), top["importance"], color=colors, alpha=0.8)
    ax.set_yticks(range(len(top)))
    ylabels = []
    for _, row in top.iterrows():
        direction = "+" if row["ds_age_rho"] > 0 else "-"
        muan_dir = "+" if row["muan_vs_ds_elderly"] > 0 else "-"
        ylabels.append(f"{row['genus']} (age{direction}, RC{muan_dir})")
    ax.set_yticklabels(ylabels, fontsize=8)
    ax.set_xlabel("Feature Importance (age model)")
    ax.set_title("Age Model Feature Importances\n"
                 "Green = makes RC look younger, Red = makes RC look older")
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color=COLORS["ds_young"], label="Makes RC look younger"),
        Patch(color=COLORS["muan"], label="Makes RC look older"),
    ], loc="lower right", fontsize=8)
    plt.tight_layout()
    fig.savefig(OUT / "figS1_feature_importance.png")
    fig.savefig(OUT / "figS1_feature_importance.pdf")
    plt.close()

    n_young = (top["young_direction"] == "younger").sum()
    print(f"    Top 20 features: {n_young} make Muan look younger, {20-n_young} older")
    return feat_imp


# ===================================================================
# 2. AGE-GENUS OVERLAP: DS age-correlated genera in Muan
# ===================================================================

def fig_age_genus_overlap(meta, genus_rel):
    print("  Analysis 2: DS age-genus correlations applied to Muan...")
    genus_clr = clr_transform(genus_rel)

    # Prevalent genera
    prevalent = (genus_rel > 0).mean(axis=0)
    prev_genera = prevalent[prevalent >= 0.10].index.tolist()

    ds_idx = meta[meta["cohort"] == "DS"].index.intersection(genus_clr.index)
    ds_age = meta.loc[ds_idx, "age"]

    groups = get_age_groups(meta)
    ds_young_idx = groups[groups == "DS young"].index.intersection(genus_clr.index)
    ds_old_idx = groups[groups == "DS elderly"].index.intersection(genus_clr.index)
    muan_idx = groups[groups == "RC elderly"].index.intersection(genus_clr.index)

    results = []
    for genus in prev_genera:
        rho, p = stats.spearmanr(ds_age, genus_clr.loc[ds_idx, genus])
        mean_ds_young = genus_clr.loc[ds_young_idx, genus].mean()
        mean_ds_old = genus_clr.loc[ds_old_idx, genus].mean()
        mean_muan = genus_clr.loc[muan_idx, genus].mean()
        results.append({
            "genus": genus,
            "ds_age_rho": rho, "ds_age_p": p,
            "mean_ds_young": mean_ds_young,
            "mean_ds_old": mean_ds_old,
            "mean_muan": mean_muan,
        })

    df = pd.DataFrame(results)
    _, df["ds_age_q"], _, _ = multipletests(df["ds_age_p"], method="fdr_bh")
    df = df.sort_values("ds_age_p")

    # Key metric: does Muan look more like DS young or DS old?
    # For each genus, compute where Muan falls relative to DS young and DS old
    df["muan_position"] = (df["mean_muan"] - df["mean_ds_young"]) / (
        df["mean_ds_old"] - df["mean_ds_young"] + 1e-10)
    # 0 = matches DS young, 1 = matches DS old, <0 = even younger than DS young

    df.to_csv(OUT / "table_age_genus_overlap.csv", index=False)

    # Plot: scatter of DS age rho vs Muan position
    sig = df[df["ds_age_q"] < 0.05].copy()
    print(f"    {len(sig)} genera significantly age-correlated in DS (FDR < 0.05)")
    print(f"    Muan position: median={sig['muan_position'].median():.2f} "
          f"(0=DS young, 1=DS old)")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: For age-increasing genera in DS, where does Muan fall?
    ax = axes[0]
    increasing = sig[sig["ds_age_rho"] > 0].sort_values("ds_age_rho", ascending=False).head(15)
    y_pos = range(len(increasing))
    for i, (_, row) in enumerate(increasing.iterrows()):
        # Plot DS young, DS old, Muan as points
        ax.plot([row["mean_ds_young"], row["mean_ds_old"]], [i, i],
                color="gray", linewidth=1, alpha=0.5)
        ax.scatter(row["mean_ds_young"], i, color=COLORS["ds_young"],
                   s=50, zorder=5, label="DS young" if i == 0 else "")
        ax.scatter(row["mean_ds_old"], i, color=COLORS["ds_old"],
                   s=50, zorder=5, label="DS elderly" if i == 0 else "")
        ax.scatter(row["mean_muan"], i, color=COLORS["muan"],
                   s=80, zorder=6, marker="D",
                   label="RC elderly" if i == 0 else "")
    ax.set_yticks(range(len(increasing)))
    ax.set_yticklabels(increasing["genus"].values, fontsize=8)
    ax.set_xlabel("Mean CLR Abundance")
    ax.set_title("A. Genera that INCREASE with age in DS\n(RC vs DS young)")
    ax.legend(fontsize=8, loc="lower right")

    # Panel B: For age-decreasing genera in DS, where does Muan fall?
    ax = axes[1]
    decreasing = sig[sig["ds_age_rho"] < 0].sort_values("ds_age_rho").head(15)
    for i, (_, row) in enumerate(decreasing.iterrows()):
        ax.plot([row["mean_ds_young"], row["mean_ds_old"]], [i, i],
                color="gray", linewidth=1, alpha=0.5)
        ax.scatter(row["mean_ds_young"], i, color=COLORS["ds_young"],
                   s=50, zorder=5, label="DS young" if i == 0 else "")
        ax.scatter(row["mean_ds_old"], i, color=COLORS["ds_old"],
                   s=50, zorder=5, label="DS elderly" if i == 0 else "")
        ax.scatter(row["mean_muan"], i, color=COLORS["muan"],
                   s=80, zorder=6, marker="D",
                   label="RC elderly" if i == 0 else "")
    ax.set_yticks(range(len(decreasing)))
    ax.set_yticklabels(decreasing["genus"].values, fontsize=8)
    ax.set_xlabel("Mean CLR Abundance")
    ax.set_title("B. Genera that DECREASE with age in DS\n(RC vs DS young)")
    ax.legend(fontsize=8, loc="lower right")


    plt.tight_layout()
    fig.savefig(OUT / "figS2_age_genus_overlap.png")
    fig.savefig(OUT / "figS2_age_genus_overlap.pdf")
    plt.close()

    # Summary stats
    increasing_all = sig[sig["ds_age_rho"] > 0]
    decreasing_all = sig[sig["ds_age_rho"] < 0]
    n_young_pattern = 0
    # For age-increasing genera: Muan < DS old (younger pattern)
    n_young_pattern += (increasing_all["mean_muan"] < increasing_all["mean_ds_old"]).sum()
    # For age-decreasing genera: Muan > DS old (younger pattern = higher than old)
    n_young_pattern += (decreasing_all["mean_muan"] > decreasing_all["mean_ds_old"]).sum()
    print(f"    {n_young_pattern}/{len(sig)} age-associated genera show younger pattern in RC")

    return df


# ===================================================================
# 3. DIET-MICROBIOME LINK (25_MB subset)
# ===================================================================

def fig_diet_microbiome(meta=None, genus_rel=None):
    """Plot Figure S4 from pre-computed Table S4 (16 foods × 11 genera = 176 tests)."""
    print("  Figure S4: Diet-microbiome heatmap from Table S4...")

    # Read pre-computed results from Table S4
    s4_path = OUT / ".." / "supplementary" / "tableS4_diet_microbiome.csv"
    if not s4_path.exists():
        s4_path = OUT / "tableS4_diet_microbiome.csv"
    if not s4_path.exists():
        print(f"    Table S4 not found at {s4_path}, skipping")
        return

    df = pd.read_csv(s4_path)
    n_min, n_max = int(df["n"].min()), int(df["n"].max())
    n_nominal = (df["p"] < 0.05).sum()
    n_sig = (df["q"] < 0.05).sum()
    print(f"    {len(df)} tests, n = {n_min}–{n_max}, {n_nominal} nominal P<0.05, {n_sig} FDR<0.05")

    # Pivot for heatmap
    pivot = df.pivot(index="genus", columns="food", values="rho")
    pivot_p = df.pivot(index="genus", columns="food", values="p")

    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(pivot.values, cmap="RdBu_r", vmin=-0.3, vmax=0.3, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9, style="italic")

    # Add rho values with significance markers
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            p_val = pivot_p.values[i, j]
            marker = ""
            if p_val < 0.001:
                marker = "***"
            elif p_val < 0.01:
                marker = "**"
            elif p_val < 0.05:
                marker = "*"
            ax.text(j, i, f"{val:.2f}{marker}", ha="center", va="center",
                    fontsize=6, color="black" if abs(val) < 0.15 else "white")

    plt.colorbar(im, ax=ax, label="Spearman rho", shrink=0.8)
    ax.set_title(f"Diet-Microbiome Correlations (2025 dietary subset, pairwise n = {n_min}–{n_max})\n"
                 f"* P<0.05, ** P<0.01, *** P<0.001; none survives FDR correction")
    plt.tight_layout()
    fig.savefig(OUT / "figS4_diet_microbiome.png", dpi=300)
    fig.savefig(OUT / "figS4_diet_microbiome.pdf")
    plt.close()
    print(f"    Saved to {OUT / 'figS4_diet_microbiome.png'}")

    return df


# ===================================================================
# 4. EXERCISE / SMOKING vs MICROBIOME AGE GAP
# ===================================================================

def fig_lifestyle_age_gap(meta, genus_rel):
    print("  Analysis 4: Lifestyle vs microbiome age gap...")
    genus_top = select_top_genera(genus_rel, 50)
    X_all = clr_transform(genus_top)

    # Train model on DS, predict all Muan
    ds_idx = meta[meta["cohort"] == "DS"].index.intersection(X_all.index)
    muan_idx = meta[meta["cohort"] == "Muan"].index.intersection(X_all.index)

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("reg", GradientBoostingRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            random_state=RANDOM_STATE, subsample=0.8)),
    ])
    pipe.fit(X_all.loc[ds_idx], meta.loc[ds_idx, "age"])
    muan_pred = pipe.predict(X_all.loc[muan_idx])
    muan_delta = muan_pred - meta.loc[muan_idx, "age"].values

    delta_df = pd.DataFrame({
        "delta_age": muan_delta,
    }, index=muan_idx)

    # Load MB questionnaire
    mb_quest = load_mb_questionnaire()
    common = delta_df.index.intersection(mb_quest.index)
    merged = delta_df.loc[common].copy()
    merged = merged.join(mb_quest.loc[common])

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel A: Exercise vs age gap
    ax = axes[0]
    if "exercise_days" in merged.columns:
        ex = merged["exercise_days"].dropna()
        ex_idx = ex.index
        rho, p = stats.spearmanr(ex, merged.loc[ex_idx, "delta_age"])
        ax.scatter(ex, merged.loc[ex_idx, "delta_age"], alpha=0.3, s=10,
                   color=COLORS["muan"])
        z = np.polyfit(ex, merged.loc[ex_idx, "delta_age"], 1)
        x_line = np.linspace(0, 7, 100)
        ax.plot(x_line, np.poly1d(z)(x_line), "k-", linewidth=2)
        ax.set_xlabel("Exercise days/week")
        ax.set_ylabel("Microbiome Age Gap (years)")
        ax.set_title(f"A. Exercise vs Age Gap\n(rho={rho:.3f}, P={p:.4f}, n={len(ex)})")
    else:
        ax.text(0.5, 0.5, "No exercise data", transform=ax.transAxes, ha="center")

    # Panel B: Smoker vs non-smoker age gap
    ax = axes[1]
    if "never_smoker" in merged.columns:
        ns = merged["never_smoker"].dropna()
        ns_idx = ns.index
        never = merged.loc[ns_idx[ns == 1], "delta_age"]
        ever = merged.loc[ns_idx[ns == 0], "delta_age"]
        if len(never) > 5 and len(ever) > 5:
            bp = ax.boxplot([never.values, ever.values],
                           labels=[f"Never\n(n={len(never)})", f"Ever\n(n={len(ever)})"],
                           patch_artist=True, widths=0.5)
            bp["boxes"][0].set_facecolor(COLORS["ds_young"])
            bp["boxes"][1].set_facecolor(COLORS["muan"])
            for b in bp["boxes"]:
                b.set_alpha(0.6)
            u, p = stats.mannwhitneyu(never, ever, alternative="two-sided")
            ax.set_ylabel("Microbiome Age Gap (years)")
            ax.set_title(f"B. Smoking vs Age Gap\n(P={p:.4f})")
    else:
        ax.text(0.5, 0.5, "No smoking data", transform=ax.transAxes, ha="center")

    # Panel C: Hypertension vs age gap
    ax = axes[2]
    if "hypertension" in merged.columns:
        htn = merged["hypertension"].dropna()
        htn_idx = htn.index
        htn_yes = merged.loc[htn_idx[htn == 1], "delta_age"]
        htn_no = merged.loc[htn_idx[htn == 0], "delta_age"]
        if len(htn_yes) > 5 and len(htn_no) > 5:
            bp = ax.boxplot([htn_no.values, htn_yes.values],
                           labels=[f"No HTN\n(n={len(htn_no)})", f"HTN\n(n={len(htn_yes)})"],
                           patch_artist=True, widths=0.5)
            bp["boxes"][0].set_facecolor(COLORS["ds_young"])
            bp["boxes"][1].set_facecolor(COLORS["muan"])
            for b in bp["boxes"]:
                b.set_alpha(0.6)
            u, p = stats.mannwhitneyu(htn_no, htn_yes, alternative="two-sided")
            ax.set_ylabel("Microbiome Age Gap (years)")
            ax.set_title(f"C. Hypertension vs Age Gap\n(P={p:.4f})")
    else:
        ax.text(0.5, 0.5, "No HTN data", transform=ax.transAxes, ha="center")


    plt.tight_layout()
    fig.savefig(OUT / "figS4_lifestyle_age_gap.png")
    fig.savefig(OUT / "figS4_lifestyle_age_gap.pdf")
    plt.close()


# ===================================================================
# 5. AGE-TRAJECTORY REVERSAL: genera that go opposite direction in Muan
# ===================================================================

def fig_age_trajectory(meta, genus_rel):
    """For top age-correlated genera in DS, plot their trajectory across
    DS age decades AND Muan age decades to see if the age trend reverses."""
    print("  Analysis 5: Age trajectory comparison DS vs Muan...")
    genus_clr = clr_transform(genus_rel)

    # Top age-correlated genera in DS
    ds_idx = meta[meta["cohort"] == "DS"].index.intersection(genus_clr.index)
    ds_age = meta.loc[ds_idx, "age"]

    prevalent = (genus_rel > 0).mean(axis=0)
    prev_genera = prevalent[prevalent >= 0.10].index.tolist()

    age_corr = []
    for genus in prev_genera:
        rho, p = stats.spearmanr(ds_age, genus_clr.loc[ds_idx, genus])
        age_corr.append({"genus": genus, "rho": rho, "p": p})
    age_df = pd.DataFrame(age_corr).sort_values("p")

    # Pick top 4 increasing and top 4 decreasing
    top_inc = age_df[age_df["rho"] > 0].head(4)["genus"].tolist()
    top_dec = age_df[age_df["rho"] < 0].head(4)["genus"].tolist()
    selected = top_inc + top_dec

    # Compute mean CLR by decade and cohort
    decade = (meta["age"] // 10 * 10).astype(int)

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.flatten()

    for i, genus in enumerate(selected):
        ax = axes[i]
        for cohort, display, color, marker in [("DS", "DS", COLORS["ds_old"], "o"),
                                              ("Muan", "RC", COLORS["muan"], "D")]:
            mask = meta["cohort"] == cohort
            c_decade = decade[mask]
            c_genus = genus_clr.loc[mask.values, genus]

            means = []
            decades_list = []
            sems = []
            for d in sorted(c_decade.unique()):
                if d < 20:
                    continue
                vals = c_genus[c_decade == d]
                if len(vals) >= 5:
                    means.append(vals.mean())
                    sems.append(vals.sem())
                    decades_list.append(d)

            if decades_list:
                ax.errorbar(decades_list, means, yerr=sems,
                           color=color, marker=marker, markersize=5,
                           linewidth=1.5, capsize=3,
                           label=display if i == 0 else "")

        rho_val = age_df[age_df["genus"] == genus]["rho"].values[0]
        direction = "+" if rho_val > 0 else "-"
        ax.set_title(f"{genus}\n(DS age rho={rho_val:.2f})", fontsize=9)
        ax.set_xlabel("Age decade")
        if i % 4 == 0:
            ax.set_ylabel("Mean CLR abundance")

    axes[0].legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(OUT / "figS5_age_trajectory.png")
    fig.savefig(OUT / "figS5_age_trajectory.pdf")
    plt.close()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("Longevity Village: Additional Analyses")
    print("=" * 50)

    meta, genus_rel = load_data()
    print(f"  DS: {(meta.cohort=='DS').sum()}, Muan: {(meta.cohort=='Muan').sum()}")

    feat_imp = fig_feature_importance(meta, genus_rel)
    age_overlap = fig_age_genus_overlap(meta, genus_rel)
    diet = fig_diet_microbiome(meta, genus_rel)
    fig_lifestyle_age_gap(meta, genus_rel)
    fig_age_trajectory(meta, genus_rel)

    print(f"\nAll saved to {OUT}")


if __name__ == "__main__":
    main()

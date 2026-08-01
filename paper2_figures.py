"""
Paper 2: Korean Medicine × Microbiome — Figures and Tables.

Usage:
    cd /home/unnot/github/health-check/microbiome_report
    python -m ml.paper2_figures
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
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from statsmodels.stats.multitest import multipletests

from ml.config import MIN_AVAILABILITY, OUTER_CV_FOLDS, RANDOM_STATE
from ml.data_join import (
    clr_transform, compute_outcomes, filter_questionnaire_cols,
    load_genus_abundance, load_kmedata, select_top_genera,
)
from ml.features import assign_blocks, build_feature_matrices, process_questionnaire
from ml.model import delong_test, make_pipeline

warnings.filterwarnings("ignore")

OUT = Path("vault/health2/results")
plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 11,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.family": "sans-serif",
})


def get_data():
    kme = load_kmedata()
    genus_rel = load_genus_abundance()
    paired = kme.index.intersection(genus_rel.index)
    ds = [x for x in paired if "_DS_" in x]
    kme = kme.loc[ds]
    genus_rel = genus_rel.loc[ds]
    outcomes = compute_outcomes(kme)
    meta = kme[["YEAR", "SEX", "AGE"]].copy()
    meta["AGE"] = pd.to_numeric(meta["AGE"], errors="coerce")
    return kme, genus_rel, outcomes, meta


# ===================================================================
# 1. KM ABLATION TEST
# ===================================================================
def km_ablation():
    """Compare M1 with vs without KM features."""
    print("  KM ablation test...")
    kme, genus_rel, outcomes, meta = get_data()

    good_cols = filter_questionnaire_cols(kme, MIN_AVAILABILITY)
    quest = kme[good_cols].copy()
    quest_processed = process_questionnaire(quest, meta)
    blocks = assign_blocks(quest_processed)

    # KM feature columns
    km_cols = set(blocks.get("korean_medicine", []) + blocks.get("sweat_temp", []))
    all_cols = list(quest_processed.columns)
    non_km_cols = [c for c in all_cols if c not in km_cols]

    print(f"    Total features: {len(all_cols)}")
    print(f"    KM features: {len(km_cols)}")
    print(f"    Non-KM features: {len(non_km_cols)}")

    X_full = quest_processed
    X_no_km = quest_processed[non_km_cols]

    targets = ["y_obesity", "y_hypertension", "y_prediabetes", "y_liver",
               "y_digestive_symptoms", "y_abnormal_stool", "y_bowel_difficulty", "y_poor_gut"]

    results = []
    for target in targets:
        y = outcomes[target].dropna()
        common = X_full.index.intersection(y.index)
        y_c = y.loc[common].astype(int)

        if y_c.nunique() < 2 or y_c.sum() < 20:
            continue

        skf = StratifiedKFold(n_splits=OUTER_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

        probs = {}
        for label, X in [("With KM", X_full), ("Without KM", X_no_km)]:
            y_prob = np.zeros(len(y_c))
            for train_idx, test_idx in skf.split(X.loc[common], y_c):
                X_train = X.loc[common].iloc[train_idx]
                y_train = y_c.iloc[train_idx]
                X_test = X.loc[common].iloc[test_idx]
                pipe = make_pipeline()
                pipe.fit(X_train, y_train)
                y_prob[test_idx] = pipe.predict_proba(X_test)[:, 1]

            auc = roc_auc_score(y_c, y_prob)
            probs[label] = y_prob
            results.append({"outcome": target, "model": label, "AUC": auc,
                            "n_features": X.shape[1]})

        # DeLong test for AUC difference
        z, p = delong_test(y_c.values, probs["With KM"], probs["Without KM"])
        results[-2]["delong_z"] = z
        results[-2]["delong_p"] = p
        results[-1]["delong_z"] = z
        results[-1]["delong_p"] = p

    df = pd.DataFrame(results)
    df.to_csv(OUT / "table1_km_ablation.csv", index=False)

    # Figure
    fig, ax = plt.subplots(figsize=(10, 5))
    pivot = df.pivot(index="outcome", columns="model", values="AUC")
    pivot.index = [o.replace("y_", "").replace("_", " ").title() for o in pivot.index]
    x = np.arange(len(pivot))
    w = 0.35
    ax.bar(x - w/2, pivot["With KM"], w, label=f"With KM ({len(km_cols)} features)", color="#2196F3")
    ax.bar(x + w/2, pivot["Without KM"], w, label=f"Without KM ({len(non_km_cols)} features)", color="#FF9800")
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=35, ha="right")
    ax.set_ylabel("AUC-ROC")
    ax.set_title("Figure 1. KM Feature Ablation: Prediction With vs Without KM Items")
    ax.legend()
    ax.set_ylim(0.5, 0.9)

    # Add delta labels with DeLong p-values
    pvals = df[df["model"] == "With KM"].set_index("outcome")["delong_p"]
    for i, outcome in enumerate(pivot.index):
        delta = pivot.loc[outcome, "With KM"] - pivot.loc[outcome, "Without KM"]
        y_pos = max(pivot.loc[outcome, "With KM"], pivot.loc[outcome, "Without KM"]) + 0.01
        raw_outcome = "y_" + outcome.lower().replace(" ", "_")
        p = pvals.get(raw_outcome, np.nan)
        p_str = f"p={p:.3f}" if not np.isnan(p) else ""
        ax.text(i, y_pos, f"{delta:+.3f}\n{p_str}", ha="center", fontsize=7, color="gray")

    plt.tight_layout()
    fig.savefig(OUT / "fig1_km_ablation.png")
    fig.savefig(OUT / "fig1_km_ablation.pdf")
    plt.close()
    return df


# ===================================================================
# 2. COLD-HEAT × MICROBIOME
# ===================================================================
def cold_heat_microbiome():
    """Cold-heat constitution vs gut genera."""
    print("  Cold-heat × microbiome...")
    kme, genus_rel, outcomes, meta = get_data()
    genus_clr = clr_transform(genus_rel)

    chq_cols = [f"CHQ{i:02d}" for i in range(1, 8)]
    chq = kme[chq_cols].apply(pd.to_numeric, errors="coerce")
    chq_mean = chq.mean(axis=1).dropna()
    terciles = pd.qcut(chq_mean, 3, labels=["Cold", "Neutral", "Hot"])

    # Prevalent genera only
    prevalent = (genus_rel > 0).mean(axis=0)
    prev_genera = prevalent[prevalent >= 0.10].index.tolist()

    results = []
    for genus in prev_genera:
        groups = []
        for label in ["Cold", "Neutral", "Hot"]:
            idx = terciles[terciles == label].index.intersection(genus_clr.index)
            groups.append(genus_clr.loc[idx, genus].values)
        if all(len(g) >= 10 for g in groups):
            stat, p = stats.kruskal(*groups)
            results.append({
                "genus": genus, "H": stat, "p_value": p,
                "mean_cold": groups[0].mean(), "mean_neutral": groups[1].mean(),
                "mean_hot": groups[2].mean(),
                "diff_hot_cold": groups[2].mean() - groups[0].mean(),
            })

    df = pd.DataFrame(results)
    _, df["q_value"], _, _ = multipletests(df["p_value"], method="fdr_bh")
    df = df.sort_values("p_value")
    df.to_csv(OUT / "table2_cold_heat_genera.csv", index=False)

    # Figure — top 20 by p-value
    top = df.head(20).sort_values("diff_hot_cold")
    fig, ax = plt.subplots(figsize=(8, 7))
    colors = ["#E53935" if d > 0 else "#1E88E5" for d in top["diff_hot_cold"]]
    ax.barh(range(len(top)), top["diff_hot_cold"], color=colors, alpha=0.8)
    ax.set_yticks(range(len(top)))
    ylabels = []
    for _, row in top.iterrows():
        marker = " *" if row["q_value"] < 0.05 else ""
        ylabels.append(f"{row['genus']}{marker}")
    ax.set_yticklabels(ylabels, fontsize=9)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("CLR Difference (Hot − Cold constitution)")
    ax.set_title("Figure 2. Top Genera by Cold-Heat Constitution\n(* FDR < 0.05)")
    plt.tight_layout()
    fig.savefig(OUT / "fig2_cold_heat_genera.png")
    fig.savefig(OUT / "fig2_cold_heat_genera.pdf")
    plt.close()
    return df


# ===================================================================
# 3. ALL SASANG SYMPTOMS × MICROBIOME
# ===================================================================
def sasang_symptoms_microbiome():
    """Test all SCATORDSYMP items against microbiome."""
    print("  Sasang symptoms × microbiome...")
    kme, genus_rel, outcomes, meta = get_data()
    genus_clr = clr_transform(genus_rel)

    prevalent = (genus_rel > 0).mean(axis=0)
    prev_genera = prevalent[prevalent >= 0.10].index.tolist()

    symp_cols = [c for c in kme.columns if c.startswith("SCATORDSYMP")]
    # Exclude high-cardinality ones
    symp_cols = [c for c in symp_cols if kme[c].nunique() <= 10 and kme[c].nunique() >= 2]

    all_results = []
    for symp_col in symp_cols:
        symp = kme[symp_col].dropna()
        common = symp.index.intersection(genus_clr.index)
        symp = symp.loc[common]
        groups_vals = symp.unique()
        if len(groups_vals) < 2:
            continue

        for genus in prev_genera:
            gv = [genus_clr.loc[symp[symp == g].index, genus].values
                  for g in sorted(groups_vals)]
            gv = [g for g in gv if len(g) >= 5]
            if len(gv) >= 2:
                stat, p = stats.kruskal(*gv)
                all_results.append({
                    "symptom": symp_col, "genus": genus, "H": stat, "p_value": p,
                })

    df = pd.DataFrame(all_results)
    if len(df) == 0:
        return df

    # FDR per symptom
    summary = []
    for symp in df["symptom"].unique():
        sub = df[df["symptom"] == symp].copy()
        _, sub["q_value"], _, _ = multipletests(sub["p_value"], method="fdr_bh")
        n_sig = (sub["q_value"] < 0.05).sum()
        top_genus = sub.nsmallest(1, "q_value")["genus"].iloc[0] if len(sub) > 0 else ""
        top_q = sub["q_value"].min()
        summary.append({
            "symptom": symp, "n_genera_tested": len(sub),
            "n_significant": n_sig, "top_genus": top_genus, "top_q": top_q,
        })

    summary_df = pd.DataFrame(summary).sort_values("n_significant", ascending=False)
    summary_df.to_csv(OUT / "table3_sasang_symptoms_summary.csv", index=False)

    # Figure — heatmap of significant counts
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_df = summary_df[summary_df["n_significant"] > 0].sort_values("n_significant")
    if len(plot_df) == 0:
        plot_df = summary_df.nsmallest(15, "top_q")
    colors = ["#E53935" if n > 0 else "#BDBDBD" for n in plot_df["n_significant"]]
    ax.barh(range(len(plot_df)), plot_df["n_significant"], color=colors)
    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(plot_df["symptom"], fontsize=8)
    ax.set_xlabel("Number of Significantly Associated Genera (FDR < 0.05)")
    ax.set_title("Figure 3. Sasang Symptom Items × Microbiome Associations")
    plt.tight_layout()
    fig.savefig(OUT / "fig3_sasang_symptoms.png")
    fig.savefig(OUT / "fig3_sasang_symptoms.pdf")
    plt.close()
    return summary_df


# ===================================================================
# 4. SHAP BY QUESTIONNAIRE BLOCK
# ===================================================================
def shap_by_block():
    """Feature importance grouped by questionnaire block."""
    print("  SHAP by block...")
    kme, genus_rel, outcomes, meta = get_data()

    good_cols = filter_questionnaire_cols(kme, MIN_AVAILABILITY)
    quest = kme[good_cols].copy()
    quest_processed = process_questionnaire(quest, meta)
    blocks = assign_blocks(quest_processed)

    targets = ["y_obesity", "y_hypertension", "y_prediabetes", "y_liver",
               "y_digestive_symptoms", "y_poor_gut"]

    block_importance = []
    for target in targets:
        y = outcomes[target].dropna()
        common = quest_processed.index.intersection(y.index)
        X = quest_processed.loc[common]
        y_c = y.loc[common].astype(int)
        if y_c.nunique() < 2 or y_c.sum() < 20:
            continue

        # Average feature importance across CV folds
        skf = StratifiedKFold(n_splits=OUTER_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        fold_imps = []
        for train_idx, test_idx in skf.split(X, y_c):
            pipe = make_pipeline()
            pipe.fit(X.iloc[train_idx], y_c.iloc[train_idx])
            fold_imps.append(pipe.named_steps["clf"].feature_importances_)
        imp = pd.Series(np.mean(fold_imps, axis=0), index=X.columns)

        for block_name, block_cols in blocks.items():
            block_cols_present = [c for c in block_cols if c in imp.index]
            if block_cols_present:
                block_imp = imp[block_cols_present].sum()
                block_importance.append({
                    "outcome": target, "block": block_name,
                    "total_importance": block_imp,
                    "n_features": len(block_cols_present),
                    "mean_importance": block_imp / len(block_cols_present),
                })

    df = pd.DataFrame(block_importance)
    df.to_csv(OUT / "table4_block_importance.csv", index=False)

    # Figure — stacked bar
    pivot = df.pivot(index="outcome", columns="block", values="total_importance").fillna(0)
    pivot.index = [o.replace("y_", "").replace("_", " ").title() for o in pivot.index]

    # Merge korean_medicine + sweat_temp into body_constitution
    if "korean_medicine" in pivot.columns and "sweat_temp" in pivot.columns:
        pivot["body_constitution"] = pivot["korean_medicine"] + pivot["sweat_temp"]
        pivot = pivot.drop(columns=["korean_medicine", "sweat_temp"])
    elif "korean_medicine" in pivot.columns:
        pivot = pivot.rename(columns={"korean_medicine": "body_constitution"})
    elif "sweat_temp" in pivot.columns:
        pivot = pivot.rename(columns={"sweat_temp": "body_constitution"})

    # Reorder blocks
    block_order = ["demographics", "lifestyle", "sleep_stress", "emotion",
                   "urination_eating", "body_constitution", "other"]
    block_order = [b for b in block_order if b in pivot.columns]
    pivot = pivot[block_order]

    colors = plt.cm.Set3(np.linspace(0, 1, len(block_order)))
    fig, ax = plt.subplots(figsize=(10, 6))
    pivot.plot(kind="barh", stacked=True, ax=ax, color=colors)
    ax.set_xlabel("Total Feature Importance (XGBoost gain)")
    ax.set_title("Fig. 1. Feature Importance by Questionnaire Block")
    ax.legend(title="Block", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    fig.savefig(OUT / "fig4_block_importance.png")
    fig.savefig(OUT / "fig4_block_importance.pdf")
    plt.close()
    return df


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("Paper 2: KM × Microbiome figures...")
    log = {}
    df1 = km_ablation()
    log["km_ablation"] = df1.to_dict(orient="records")
    df2 = cold_heat_microbiome()
    log["cold_heat_n_sig"] = int((df2["q_value"] < 0.05).sum())
    df3 = sasang_symptoms_microbiome()
    if len(df3) > 0:
        log["sasang_symptoms_with_sig"] = int((df3["n_significant"] > 0).sum())
    df4 = shap_by_block()
    log["block_importance_outcomes"] = sorted(df4["outcome"].unique().tolist())
    with open(OUT / "summary_log.json", "w") as f:
        json.dump(log, f, indent=2, default=str)
    print(f"\nAll saved to {OUT}")


if __name__ == "__main__":
    main()

"""
Longevity Village Analysis: sensitivity analyses (Table S5).

All analyses reuse the fixed set of 153 genera (>=10% prevalence across all
2,084 samples) with BH-FDR across those 153 tests, Mann-Whitney U on CLR
(pseudocount 1e-6), RC vs DS adults aged 60+ unless stated:

  (i)   age restriction to 60-69 (+ Bray-Curtis PERMANOVA)
  (iii) sex-stratified tests and cohort x sex interaction (OLS on CLR)
  (iv)  restriction to the sampling years shared by both cohorts (2022, 2024)
  (v)   rarefaction to 10,000 genus-classified reads without replacement
        (one random draw; counts vary by a few genera between seeds)
  (vi)  exclusion of the non-rural RC sites (Geumnam-dong, Nampyeong-eup)

(ii) ANCOM-BC2 is in longevity_ancombc2.py.

Usage (set paths in ml/longevity_config.py first):
    cd /path/to/kme-microbiome-analysis
    python -m ml.longevity_sensitivity
"""
import json

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests

from ml.config import RANDOM_STATE
from ml.data_join import clr_transform
from ml.longevity_config import (
    LONGEVITY_ASV_TABLE, LONGEVITY_OUT, LONGEVITY_TAXONOMY, SAMPLE_GROUPS,
)
from ml.longevity_fig3 import GENERA as FIG3_GENERA
from ml.longevity_muan import _bray_curtis, _permanova, get_age_groups, load_all_data

OUT = LONGEVITY_OUT
RAREFY_DEPTH = 10_000
SHARED_YEARS = ("22", "24")
NON_RURAL_SITES = ("나주시 금남동", "나주시 남평읍")  # Geumnam-dong (urban), Nampyeong-eup (town)

# Key genera: the 14 shown in Fig. 3, plus Ligilactobacillus (reported for men in Table S5)
KEY_GENERA = FIG3_GENERA + ["Ligilactobacillus"]


# ===================================================================
# SHARED HELPERS
# ===================================================================

def prevalent_genera(genus_rel, min_prev=0.10):
    """Fixed genus set: >=10% prevalence across all samples."""
    prevalence = (genus_rel > 0).mean(axis=0)
    return prevalence[prevalence >= min_prev].index.tolist()


def mw_da(genus_clr, rc_idx, ds_idx, genera):
    """Mann-Whitney RC vs DS on CLR for a fixed genus list, BH-FDR across the list."""
    rows = []
    for genus in genera:
        if genus not in genus_clr.columns:
            continue
        r, d = genus_clr.loc[rc_idx, genus].values, genus_clr.loc[ds_idx, genus].values
        _, p = stats.mannwhitneyu(r, d, alternative="two-sided")
        delta = (np.sum(r[:, None] > d[None, :])
                 - np.sum(r[:, None] < d[None, :])) / (len(r) * len(d))
        rows.append({"genus": genus, "clr_diff": r.mean() - d.mean(),
                     "p_value": p, "cliffs_delta": delta})
    df = pd.DataFrame(rows)
    _, df["q_value"], _, _ = multipletests(df["p_value"], method="fdr_bh")
    return df.sort_values("p_value").reset_index(drop=True)


def summarize(name, df, primary_sig, n_ds, n_rc):
    sig = set(df.loc[df["q_value"] < 0.05, "genus"])
    q = df.set_index("genus")["q_value"]
    return {
        "analysis": name, "ds_n": n_ds, "rc_n": n_rc,
        "n_significant": len(sig),
        "overlap_with_primary": len(sig & primary_sig),
        "key_q": {g: float(q[g]) for g in KEY_GENERA if g in q.index},
    }


def load_genus_counts():
    """Genus-level read counts (samples x genera) from the combined run."""
    asv = pd.read_csv(LONGEVITY_ASV_TABLE, sep="\t", index_col=0)
    asv = asv.drop(columns=[c for c in asv.columns if c == "sequence"])
    tax = pd.read_csv(LONGEVITY_TAXONOMY, sep="\t", index_col=0)
    common = asv.index.intersection(tax.index)
    asv = asv.loc[common]
    asv["genus_label"] = tax.loc[common, "Genus"].fillna("Unclassified")
    return asv.groupby("genus_label").sum().T


def rarefy(counts, depth, seed):
    """Subsample each row to `depth` reads without replacement; drop shallower rows."""
    rng = np.random.default_rng(seed)
    keep = counts.sum(axis=1) >= depth
    out = np.vstack([rng.multivariate_hypergeometric(row.astype(np.int64), depth)
                     for row in counts.loc[keep].values])
    return pd.DataFrame(out, index=counts.index[keep], columns=counts.columns)


# ===================================================================
# ANALYSES
# ===================================================================

def age_restricted(meta, genus_rel, genus_clr, genera, primary_sig):
    """(i) RC and DS both restricted to ages 60-69."""
    rc = meta.index[(meta["cohort"] == "Muan") & meta["age"].between(60, 69)]
    ds = meta.index[(meta["cohort"] == "DS") & meta["age"].between(60, 69)]
    df = mw_da(genus_clr, rc, ds, genera)

    idx = rc.append(ds)
    labels = meta.loc[idx, "cohort"].values
    np.random.seed(RANDOM_STATE)
    f_stat, p_perm = _permanova(_bray_curtis(genus_rel.loc[idx]), labels)
    n, k = len(idx), 2
    r2 = f_stat * (k - 1) / (f_stat * (k - 1) + (n - k))

    res = summarize("(i) Age restriction 60-69", df, primary_sig, len(ds), len(rc))
    res.update({"permanova_F": f_stat, "permanova_P": p_perm, "permanova_R2_pct": 100 * r2})
    return res, df


def sex_stratified(meta, genus_clr, groups, genera, primary_df, primary_sig):
    """(iii) Separate tests in women and men, plus cohort x sex interaction
    for the key genera enriched in RC."""
    results, tables = [], {}
    for sex, label in [("F", "women"), ("M", "men")]:
        rc = groups.index[(groups == "RC elderly") & (meta["sex"] == sex)]
        ds = groups.index[(groups == "DS elderly") & (meta["sex"] == sex)]
        df = mw_da(genus_clr, rc, ds, genera)
        results.append(summarize(f"(iii) Sex-stratified ({label})", df, primary_sig, len(ds), len(rc)))
        tables[label] = df

    # Interaction: CLR ~ cohort * sex among RC + DS 60+ with known sex
    idx = groups.index[groups.isin(["RC elderly", "DS elderly"]) & meta["sex"].isin(["F", "M"])]
    base = pd.DataFrame({"cohort": (groups.loc[idx] == "RC elderly").astype(int),
                         "sex": meta.loc[idx, "sex"]})
    clr_diff = primary_df.set_index("genus")["clr_diff"]
    enriched = [g for g in KEY_GENERA if clr_diff.get(g, 0) > 0]
    interaction = {}
    for genus in enriched:
        d = base.assign(y=genus_clr.loc[idx, genus].values)
        fit = smf.ols("y ~ cohort * C(sex)", data=d).fit()
        interaction[genus] = float(fit.pvalues["cohort:C(sex)[T.M]"])
    results[0]["interaction_P"] = interaction
    return results, tables


def shared_years(meta, genus_clr, groups, genera, primary_sig):
    """(iv) Only samples from sampling years present in both cohorts."""
    year = pd.Series(meta.index.str.split("_").str[0], index=meta.index)
    in_years = year.isin(SHARED_YEARS)
    rc = groups.index[(groups == "RC elderly") & in_years]
    ds = groups.index[(groups == "DS elderly") & in_years]
    df = mw_da(genus_clr, rc, ds, genera)
    return summarize("(iv) Shared years (2022+2024)", df, primary_sig, len(ds), len(rc)), df


def rarefied(meta, groups, genera, primary_df, primary_sig):
    """(v) Rarefy genus-classified reads to 10,000, then repeat the test."""
    counts = load_genus_counts().drop(columns="Unclassified", errors="ignore")
    counts = counts.loc[counts.index.intersection(meta.index)]
    rare = rarefy(counts, RAREFY_DEPTH, RANDOM_STATE)
    rare_clr = clr_transform(rare.div(rare.sum(axis=1), axis=0))

    rc = groups.index[(groups == "RC elderly")].intersection(rare.index)
    ds = groups.index[(groups == "DS elderly")].intersection(rare.index)
    df = mw_da(rare_clr, rc, ds, [g for g in genera if g != "Unclassified"])

    res = summarize("(v) Rarefaction (10,000 reads)", df, primary_sig, len(ds), len(rc))
    lost = primary_sig - set(df.loc[df["q_value"] < 0.05, "genus"])
    delta = primary_df.set_index("genus").loc[sorted(lost), "cliffs_delta"].abs()
    res.update({"rc_excluded": int((groups == "RC elderly").sum() - len(rc)),
                "n_lost": len(lost), "lost_median_abs_delta": float(delta.median()),
                "lost_genera": sorted(lost)})
    return res, df


def non_rural_excluded(meta, genus_clr, groups, genera, primary_df, primary_sig):
    """(vi) Drop RC participants from the urban/town sites."""
    region = pd.read_csv(SAMPLE_GROUPS, index_col="sample_id")["region"]
    rural = ~region.reindex(meta.index).isin(NON_RURAL_SITES)
    rc = groups.index[(groups == "RC elderly") & rural]
    ds = groups.index[groups == "DS elderly"]
    df = mw_da(genus_clr, rc, ds, genera)
    res = summarize("(vi) Non-rural sites excluded", df, primary_sig, len(ds), len(rc))
    # Effect-size stability for the primary 70
    m = primary_df[primary_df["genus"].isin(primary_sig)].merge(df, on="genus", suffixes=("", "_sub"))
    res["clr_diff_corr_primary70"] = float(np.corrcoef(m["clr_diff"], m["clr_diff_sub"])[0, 1])
    res["median_abs_clr_diff_change"] = float((m["clr_diff_sub"] - m["clr_diff"]).abs().median())
    return res, df


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    meta, genus_rel, _ = load_all_data()
    groups = get_age_groups(meta)
    genus_clr = clr_transform(genus_rel)
    genera = prevalent_genera(genus_rel)

    primary = mw_da(genus_clr, groups.index[groups == "RC elderly"],
                    groups.index[groups == "DS elderly"], genera)
    primary_sig = set(primary.loc[primary["q_value"] < 0.05, "genus"])
    print(f"  Primary: {len(genera)} genera, {len(primary_sig)} significant")

    summary = [summarize("Primary (DS 60+ vs RC)", primary, primary_sig,
                         int((groups == "DS elderly").sum()), int((groups == "RC elderly").sum()))]
    tables = {}

    res, tables["age_60_69"] = age_restricted(meta, genus_rel, genus_clr, genera, primary_sig)
    summary.append(res)
    res, sex_tables = sex_stratified(meta, genus_clr, groups, genera, primary, primary_sig)
    summary.extend(res)
    tables.update({f"sex_{k}": v for k, v in sex_tables.items()})
    res, tables["shared_years"] = shared_years(meta, genus_clr, groups, genera, primary_sig)
    summary.append(res)
    res, tables["rarefied"] = rarefied(meta, groups, genera, primary, primary_sig)
    summary.append(res)
    res, tables["non_rural_excluded"] = non_rural_excluded(meta, genus_clr, groups, genera, primary, primary_sig)
    summary.append(res)

    for name, df in tables.items():
        df.to_csv(OUT / f"sensitivity_{name}.csv", index=False)
    with open(OUT / "sensitivity_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    for s in summary:
        print(f"  {s['analysis']}: DS n={s['ds_n']}, RC n={s['rc_n']}, "
              f"{s['n_significant']} significant, {s['overlap_with_primary']} of {len(primary_sig)} primary")
    print(f"  Saved to {OUT}")


if __name__ == "__main__":
    main()

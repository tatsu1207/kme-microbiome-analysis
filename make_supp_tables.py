"""
Generate supplementary tables for questionnaire data.

Usage:
    cd /home/unnot/github/health-check/microbiome_report
    python -m ml.make_supp_tables
"""
import numpy as np
import pandas as pd
from pathlib import Path

from ml.config import MIN_AVAILABILITY
from ml.data_join import (
    compute_outcomes,
    filter_questionnaire_cols,
    load_genus_abundance,
    load_kmedata,
)
from ml.features import assign_blocks, process_questionnaire

RESULTS_DIR = Path("vault/health/results")


def get_ds_data():
    kme = load_kmedata()
    genus = load_genus_abundance()
    paired = kme.index.intersection(genus.index)
    ds = [x for x in paired if "_DS_" in x]
    return kme.loc[ds], genus.loc[ds]


def table_s2_demographics():
    """Table S2: Cohort demographics and clinical characteristics."""
    print("  Table S2: Demographics...")
    kme, genus = get_ds_data()
    outcomes = compute_outcomes(kme)
    n = len(kme)

    rows = []

    # --- Demographics ---
    rows.append({"Category": "Demographics", "Variable": "N", "Value": str(n)})

    sex = kme["SEX"]
    rows.append({"Category": "", "Variable": "Female, n (%)",
                 "Value": f"{(sex == 'F').sum()} ({(sex == 'F').mean()*100:.1f}%)"})
    rows.append({"Category": "", "Variable": "Male, n (%)",
                 "Value": f"{(sex == 'M').sum()} ({(sex == 'M').mean()*100:.1f}%)"})

    age = pd.to_numeric(kme["AGE"], errors="coerce")
    rows.append({"Category": "", "Variable": "Age, years (mean +/- SD)",
                 "Value": f"{age.mean():.1f} +/- {age.std():.1f}"})
    rows.append({"Category": "", "Variable": "Age range",
                 "Value": f"{age.min():.0f} - {age.max():.0f}"})

    ht = pd.to_numeric(kme["HEIGHT"], errors="coerce")
    rows.append({"Category": "", "Variable": "Height, cm (mean +/- SD)",
                 "Value": f"{ht.mean():.1f} +/- {ht.std():.1f}"})

    # Education
    ed_map = {"1": "None", "2": "Elementary", "3": "Middle school",
              "4": "High school", "5": "Associate", "6": "Bachelor", "7": "Graduate"}
    ed = kme.get("EDLEVEL")
    if ed is not None:
        ed_counts = ed.value_counts()
        rows.append({"Category": "Education", "Variable": "", "Value": ""})
        for code in sorted(ed_map.keys()):
            if code in ed_counts.index:
                cnt = ed_counts[code]
                rows.append({"Category": "", "Variable": f"  {ed_map[code]}",
                             "Value": f"{cnt} ({cnt/n*100:.1f}%)"})

    # Marital status
    mar_map = {"SG": "Single", "MG": "Married", "DV": "Divorced", "BM": "Bereaved"}
    mar = kme.get("MARISTAT")
    if mar is not None:
        rows.append({"Category": "Marital status", "Variable": "", "Value": ""})
        for code, label in mar_map.items():
            cnt = (mar == code).sum()
            if cnt > 0:
                rows.append({"Category": "", "Variable": f"  {label}",
                             "Value": f"{cnt} ({cnt/n*100:.1f}%)"})

    # Employment
    emp_map = {"HW": "Housework", "PJ": "Professional", "OW": "Office worker",
               "ETC": "Other", "SP": "Self-employed (professional)",
               "MP": "Manual/production", "SL": "Sales/service",
               "TS": "Technical/skilled", "MO": "Management/officer",
               "FF": "Farming/fishing", "ST": "Student", "UN": "Unemployed",
               "SW": "Social worker", "CS": "Civil servant"}
    emp = kme.get("EMPJOB")
    if emp is not None:
        rows.append({"Category": "Employment", "Variable": "", "Value": ""})
        for code in emp.value_counts().index:
            cnt = (emp == code).sum()
            label = emp_map.get(code, code)
            rows.append({"Category": "", "Variable": f"  {label}",
                         "Value": f"{cnt} ({cnt/n*100:.1f}%)"})

    # Handedness
    hand_map = {"RH": "Right-handed", "LH": "Left-handed", "BH": "Ambidextrous"}
    hand = kme.get("HANDDOM")
    if hand is not None:
        rows.append({"Category": "Handedness", "Variable": "", "Value": ""})
        for code, label in hand_map.items():
            cnt = (hand == code).sum()
            if cnt > 0:
                rows.append({"Category": "", "Variable": f"  {label}",
                             "Value": f"{cnt} ({cnt/n*100:.1f}%)"})

    # --- Lifestyle ---
    rows.append({"Category": "Lifestyle", "Variable": "", "Value": ""})

    cig = kme.get("CIGARYN")
    if cig is not None:
        smokers = (cig == "1").sum()
        rows.append({"Category": "", "Variable": "Current/former smoker, n (%)",
                     "Value": f"{smokers} ({smokers/n*100:.1f}%)"})

    alc = kme.get("ALCOHOLYN")
    if alc is not None:
        drinkers = (alc == "1").sum()
        rows.append({"Category": "", "Variable": "Current drinker, n (%)",
                     "Value": f"{drinkers} ({drinkers/n*100:.1f}%)"})

    # --- Clinical measurements ---
    rows.append({"Category": "Clinical measurements", "Variable": "", "Value": ""})
    for col, label, unit in [
        ("PBF", "Body fat percentage", "%"),
        ("SYSBP", "Systolic blood pressure", "mmHg"),
        ("DIABP", "Diastolic blood pressure", "mmHg"),
        ("GLUC", "Fasting glucose", "mg/dL"),
        ("ALT", "ALT", "U/L"),
        ("GGT", "GGT", "U/L"),
    ]:
        v = pd.to_numeric(kme.get(col), errors="coerce")
        valid = v.dropna()
        rows.append({"Category": "", "Variable": f"  {label}, {unit} (mean +/- SD)",
                     "Value": f"{valid.mean():.1f} +/- {valid.std():.1f}"})

    # --- Sleep & Stress ---
    rows.append({"Category": "Sleep & Stress", "Variable": "", "Value": ""})
    psqi = pd.to_numeric(kme.get("PSQITOTAL"), errors="coerce").dropna()
    rows.append({"Category": "", "Variable": "PSQI total (mean +/- SD)",
                 "Value": f"{psqi.mean():.1f} +/- {psqi.std():.1f}"})
    poor_sleep = (psqi > 5).sum()
    rows.append({"Category": "", "Variable": "Poor sleep (PSQI > 5), n (%)",
                 "Value": f"{poor_sleep} ({poor_sleep/len(psqi)*100:.1f}%)"})

    pss = pd.to_numeric(kme.get("PSS10"), errors="coerce").dropna()
    rows.append({"Category": "", "Variable": "PSS-10 mean score (mean +/- SD)",
                 "Value": f"{pss.mean():.2f} +/- {pss.std():.2f}"})

    # --- Korean Medicine ---
    rows.append({"Category": "Korean Medicine", "Variable": "", "Value": ""})
    chq_cols = [f"CHQ{i:02d}" for i in range(1, 8)]
    chq = kme[chq_cols].apply(pd.to_numeric, errors="coerce")
    chq_mean = chq.mean(axis=1).dropna()
    rows.append({"Category": "", "Variable": "CHQ cold-heat score (mean +/- SD)",
                 "Value": f"{chq_mean.mean():.2f} +/- {chq_mean.std():.2f}"})

    terciles = pd.qcut(chq_mean, 3, labels=["Cold", "Neutral", "Hot"])
    for label in ["Cold", "Neutral", "Hot"]:
        cnt = (terciles == label).sum()
        rows.append({"Category": "", "Variable": f"  {label} constitution, n (%)",
                     "Value": f"{cnt} ({cnt/len(terciles)*100:.1f}%)"})

    # --- Outcome prevalence ---
    rows.append({"Category": "Outcome prevalence", "Variable": "", "Value": ""})
    outcome_labels = {
        "y_obesity": "Obesity (elevated body fat)",
        "y_hypertension": "Hypertension (SBP>=130 or DBP>=85)",
        "y_prediabetes": "Prediabetes (glucose>=100)",
        "y_liver": "Liver dysfunction (ALT>40 or GGT>60)",
        "y_digestive_symptoms": "Digestive symptoms",
        "y_multi_digestive": "Multiple digestive symptoms (>=2)",
        "y_abnormal_stool": "Abnormal stool form (BSS 1-2 or 6-7)",
        "y_bowel_difficulty": "Bowel difficulty",
        "y_poor_gut": "Poor gut composite (>=2)",
    }
    for col, label in outcome_labels.items():
        v = outcomes[col].dropna()
        pos = v.sum()
        rows.append({"Category": "", "Variable": f"  {label}",
                     "Value": f"{pos:.0f}/{len(v)} ({v.mean()*100:.1f}%)"})

    # Multimorbidity
    mm = outcomes["y_multimorbidity"].dropna()
    rows.append({"Category": "", "Variable": "  Multimorbidity (>=2 metabolic)",
                 "Value": f"{(mm >= 2).sum():.0f}/{len(mm)} ({(mm >= 2).mean()*100:.1f}%)"})

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "tableS2_demographics.csv", index=False)
    return df


def table_s3_questionnaire_blocks():
    """Table S3: Questionnaire feature blocks and example items."""
    print("  Table S3: Questionnaire blocks...")
    kme, _ = get_ds_data()

    good_cols = filter_questionnaire_cols(kme, MIN_AVAILABILITY)
    quest = kme[good_cols].copy()
    meta = kme[["YEAR", "SEX", "AGE"]].copy()
    meta["AGE"] = pd.to_numeric(meta["AGE"], errors="coerce")
    quest_processed = process_questionnaire(quest, meta)
    blocks = assign_blocks(quest_processed)

    rows = []
    block_descriptions = {
        "demographics": "Age, sex, height, education, marital status, employment, handedness",
        "lifestyle": "Smoking status, alcohol consumption, physical activity (IPAQ)",
        "sleep_stress": "Pittsburgh Sleep Quality Index (PSQI, 7 components + total), "
                        "Perceived Stress Scale (PSS-10), sleep duration/quality",
        "emotion": "Comprehensive State Emotion Inventory-Short (CSEIS): "
                   "anxiety, depression, anger, fear, thought interference, social avoidance, happiness",
        "urination_eating": "Urination frequency/urgency/nocturia, meal regularity, "
                           "appetite, eating speed, food preferences",
        "allergy": "Allergy history, atopic conditions",
        "disease_history": "Self-reported hypertension, diabetes, dyslipidemia history",
        "korean_medicine": "Sasang constitution ordinal symptoms (SCATORDSYMP, 23 items): "
                          "appetite, digestion, stool, sweating, body temperature, "
                          "fatigue, pain patterns. Cold-heat questionnaire (CHQ, 7 items). "
                          "Sasang Digestive Function Inventory (SDFI). "
                          "Sasang Personality Questionnaire (SPQ). Sasang Urination Index (SUI).",
        "sweat_temp": "Perspiration patterns, body temperature tendencies, "
                      "cold/heat sensitivity (SWEAT items, CHQ overlap)",
        "other": "Remaining questionnaire items not assigned to specific blocks: "
                 "additional symptom items, physical measurements, other scales",
    }

    for block, cols in blocks.items():
        if len(cols) == 0:
            continue
        rows.append({
            "Block": block.replace("_", " ").title(),
            "N features": len(cols),
            "Description": block_descriptions.get(block, ""),
            "Example columns": ", ".join(cols[:5]) + ("..." if len(cols) > 5 else ""),
        })

    # Total
    total = sum(len(cols) for cols in blocks.values())
    rows.append({
        "Block": "TOTAL",
        "N features": total,
        "Description": "",
        "Example columns": "",
    })

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "tableS3_questionnaire_blocks.csv", index=False)
    return df


def table_s4_microbiome_summary():
    """Table S4: Microbiome data summary."""
    print("  Table S4: Microbiome summary...")
    kme, genus = get_ds_data()

    rows = []

    rows.append({"Variable": "Sequencing method", "Value": "16S rRNA V3-V4"})
    rows.append({"Variable": "Bioinformatic pipeline", "Value": "DADA2"})
    rows.append({"Variable": "Total ASVs detected", "Value": "35,207"})
    rows.append({"Variable": "Total genera detected", "Value": str(genus.shape[1])})
    rows.append({"Variable": "Genera with >=10% prevalence", "Value": str((
        (genus > 0).mean(axis=0) >= 0.10).sum())})

    obs = (genus > 0).sum(axis=1)
    rows.append({"Variable": "Observed genera per sample (mean +/- SD)",
                 "Value": f"{obs.mean():.1f} +/- {obs.std():.1f}"})

    # Shannon on genus relative abundance
    def shannon(row):
        p = row[row > 0]
        return -np.sum(p * np.log(p))
    shan = genus.apply(shannon, axis=1)
    rows.append({"Variable": "Shannon diversity (mean +/- SD)",
                 "Value": f"{shan.mean():.3f} +/- {shan.std():.3f}"})

    # Top 10 genera
    mean_abund = genus.mean().sort_values(ascending=False)
    rows.append({"Variable": "", "Value": ""})
    rows.append({"Variable": "Top 10 genera by mean relative abundance", "Value": ""})
    for i, (g, v) in enumerate(mean_abund.head(10).items()):
        rows.append({"Variable": f"  {i+1}. {g}", "Value": f"{v:.1%}"})

    # Enterotype distribution
    rows.append({"Variable": "", "Value": ""})
    rows.append({"Variable": "Enterotype distribution", "Value": ""})
    rows.append({"Variable": "  ET1 (Prevotella-dominant)", "Value": "347 (44.8%)"})
    rows.append({"Variable": "  ET2 (Bacteroides-dominant)", "Value": "427 (55.2%)"})

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "tableS4_microbiome_summary.csv", index=False)
    return df


def table_s5_outcome_by_sex():
    """Table S5: Outcome prevalence stratified by sex."""
    print("  Table S5: Outcomes by sex...")
    kme, _ = get_ds_data()
    outcomes = compute_outcomes(kme)
    sex = kme["SEX"]

    outcome_labels = {
        "y_obesity": "Obesity",
        "y_hypertension": "Hypertension",
        "y_prediabetes": "Prediabetes",
        "y_liver": "Liver dysfunction",
        "y_digestive_symptoms": "Digestive symptoms",
        "y_multi_digestive": "Multiple digestive symptoms",
        "y_abnormal_stool": "Abnormal stool",
        "y_bowel_difficulty": "Bowel difficulty",
        "y_poor_gut": "Poor gut composite",
    }

    rows = []
    for col, label in outcome_labels.items():
        y = outcomes[col].dropna()
        for s, s_label in [("M", "Male"), ("F", "Female")]:
            idx = sex[sex == s].index.intersection(y.index)
            ys = y.loc[idx]
            pos = ys.sum()
            rows.append({
                "Outcome": label,
                "Sex": s_label,
                "n": len(ys),
                "Cases": f"{pos:.0f}",
                "Prevalence (%)": f"{ys.mean()*100:.1f}",
            })

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "tableS5_outcomes_by_sex.csv", index=False)
    return df


def table_s6_alpha_diversity():
    """Table S6: Alpha diversity by outcome (full table)."""
    print("  Table S6: Alpha diversity...")
    alpha_path = Path("ml/output/analysis/alpha_diversity.csv")
    if alpha_path.exists():
        df = pd.read_csv(alpha_path)
        df["Outcome"] = df["outcome"].str.replace("y_", "").str.replace("_", " ").str.title()
        df["Metric"] = df["metric"].str.title()
        table = df[["Outcome", "Metric", "mean_disease", "mean_healthy",
                     "difference", "p_value", "q_value", "significant"]].copy()
        table.columns = ["Outcome", "Metric", "Disease Mean", "Healthy Mean",
                         "Difference", "p-value", "FDR q-value", "Significant"]
        for col in ["Disease Mean", "Healthy Mean", "Difference"]:
            table[col] = table[col].apply(lambda x: f"{x:.4f}")
        for col in ["p-value", "FDR q-value"]:
            table[col] = table[col].apply(lambda x: f"{x:.4f}")
        table.to_csv(RESULTS_DIR / "tableS6_alpha_diversity.csv", index=False)
    return


def table_s7_per_sample_sequencing():
    """Table S7: Per-sample sequencing statistics (reads, ASVs, diversity)."""
    print("  Table S7: Per-sample sequencing stats...")
    kme, genus_rel = get_ds_data()

    # Read raw ASV table for read counts and ASV counts per sample
    from ml.config import ASV_TABLE
    asv = pd.read_csv(ASV_TABLE, sep="\t", index_col=0)
    # ASV table: rows=ASVs, cols=samples
    ds_samples = [s for s in asv.columns if "_DS_" in s]
    ds_samples = [s for s in ds_samples if s in genus_rel.index]
    asv_ds = asv[ds_samples]

    # Per-sample stats
    total_reads = asv_ds.sum(axis=0)
    n_asvs = (asv_ds > 0).sum(axis=0)
    n_genera = (genus_rel.loc[ds_samples] > 0).sum(axis=1)

    # Alpha diversity on genus relative abundance
    def shannon(row):
        p = row[row > 0]
        return -np.sum(p * np.log(p))

    def simpson(row):
        return 1 - np.sum(row ** 2)

    shannon_vals = genus_rel.loc[ds_samples].apply(shannon, axis=1)
    simpson_vals = genus_rel.loc[ds_samples].apply(simpson, axis=1)

    per_sample = pd.DataFrame({
        "sample_id": ds_samples,
        "total_reads": total_reads.values,
        "observed_ASVs": n_asvs.values,
        "observed_genera": n_genera.values,
        "shannon": shannon_vals.values,
        "simpson": simpson_vals.values,
    })
    per_sample.to_csv(RESULTS_DIR / "tableS7_per_sample_stats.csv", index=False)

    # Print summary
    print(f"    Samples: {len(per_sample)}")
    print(f"    Reads: median={total_reads.median():.0f}, "
          f"mean={total_reads.mean():.0f}, "
          f"range={total_reads.min():.0f}-{total_reads.max():.0f}")
    print(f"    ASVs/sample: median={n_asvs.median():.0f}, "
          f"mean={n_asvs.mean():.1f}, "
          f"range={n_asvs.min()}-{n_asvs.max()}")
    print(f"    Genera/sample: median={n_genera.median():.0f}, "
          f"mean={n_genera.mean():.1f}")
    print(f"    Shannon: mean={shannon_vals.mean():.3f} +/- {shannon_vals.std():.3f}")
    print(f"    Simpson: mean={simpson_vals.mean():.3f} +/- {simpson_vals.std():.3f}")

    return per_sample


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("Generating supplementary tables...")
    table_s2_demographics()
    table_s3_questionnaire_blocks()
    table_s4_microbiome_summary()
    table_s5_outcome_by_sex()
    table_s6_alpha_diversity()
    table_s7_per_sample_sequencing()
    print(f"\nAll supplementary tables saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()

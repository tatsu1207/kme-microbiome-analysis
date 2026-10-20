"""
Longevity Village Analysis: Table 3 (diet and lifestyle of the RC).

  Diet      2025 food frequency questionnaire (all respondents of the 2025
            wave). Frequency codes 1-8 (1 = rarely ... 4 = once a week ...
            7 = once a day, 8 = twice a day or more): "at least weekly" =
            codes 4-8, "daily or more" = codes 7-8, "twice daily or more" =
            code 8. n = respondents with a valid answer for that food group.
  Smoking   2022 wave, never smokers, RC participants in the microbiome cohort.
  Alcohol   2022 wave, RC participants in the microbiome cohort with a valid
            answer: abstainer (code 1), less than monthly (codes 1-2),
            weekly or more (codes 5-7).
  Exercise  2022-2024 waves, days per week with >= 10 min of exercise or
            physical work, RC participants in the microbiome cohort; mean and
            per-wave share exercising daily.

Completion rate = n / 1,256 (RC microbiome cohort).

Usage (set paths in ml/longevity_config.py first):
    cd /path/to/kme-microbiome-analysis
    python -m ml.longevity_table3
"""
import pandas as pd

from ml.longevity_config import LONGEVITY_OUT, MB_QUESTIONNAIRE_DIR
from ml.longevity_muan import load_all_data

OUT = LONGEVITY_OUT

# 2025 FFQ: food group -> "overall frequency" column
FFQ_ITEMS = {
    "Kimchi": "김치 전체 섭취 횟수",
    "Vegetables": "채소 전체 섭취 횟수",
    "Namul (cooked vegetables)": "나물 전체 섭취 횟수",
    "Seaweed": "해조류 전체 섭취 횟수",
    "Dairy (milk, yogurt)": "유제품 전체 섭취 횟수",
    "Egg dishes": "달걀 요리 전체 섭취 횟수",
    "Fish": "생선 전체 섭취 횟수",
    "Jangajji/jeotgal": "장아찌/젓갈 전체 섭취 횟수",
    "Meat": "육류 전체 섭취 횟수",
    "Semi-prepared/processed food": "반조리 요리 전체 섭취 횟수",
}
WEEKLY, DAILY, TWICE_DAILY = 4, 7, 8

ALCOHOL_COL = "9. 최근 1년 동안 술을 얼마나 자주 드셨습니까?"
SMOKING_NEVER = "피운 적 없"  # "never smoked" answer to the lifetime-amount question
EXERCISE_KEYS = ("운동", "스포츠")


def normalize_id(raw):
    """22-MB-0001 -> 22_MB_001"""
    parts = str(raw).replace("-", "_").split("_")
    return f"{parts[0]}_{parts[1]}_{int(parts[2]):03d}" if len(parts) == 3 else str(raw)


def read_sheet(year, keyword):
    path = MB_QUESTIONNAIRE_DIR / f"{year}.xlsx"
    sheet = next(s for s in pd.ExcelFile(path).sheet_names if keyword in s)
    df = pd.read_excel(path, sheet_name=sheet)
    df.index = df["연구대상자ID"].map(normalize_id)
    return df


def row(section, label, n, value, n_total):
    return {"section": section, "characteristic": label, "n": int(n),
            "value": value, "completion_pct": round(100 * n / n_total, 1)}


def diet_rows(n_total):
    ffq = read_sheet(25, "식생")
    rows = []
    for food, col in FFQ_ITEMS.items():
        codes = pd.to_numeric(ffq[col], errors="coerce").dropna()
        n = len(codes)
        if food == "Kimchi":
            rows.append(row("Diet (2025 FFQ)", "Kimchi, twice daily or more (code 8)", n,
                            round(100 * (codes >= TWICE_DAILY).mean(), 1), n_total))
            rows.append(row("Diet (2025 FFQ)", "Kimchi, daily or more (codes 7-8)", n,
                            round(100 * (codes >= DAILY).mean(), 1), n_total))
        if food == "Meat":
            rows.append(row("Diet (2025 FFQ)", "Meat, daily or more (codes 7-8)", n,
                            round(100 * (codes >= DAILY).mean(), 1), n_total))
        else:
            rows.append(row("Diet (2025 FFQ)", f"{food}, at least weekly", n,
                            round(100 * (codes >= WEEKLY).mean(), 1), n_total))
    return rows


def lifestyle_rows(rc_ids, n_total):
    life22 = read_sheet(22, "건강생활").reindex(rc_ids).dropna(how="all")

    smoke_col = next(c for c in life22.columns if "담배" in c and "전자" not in c)
    smoke = life22[smoke_col].dropna().astype(str)
    rows = [row("Smoking (2022)", "Never smokers", len(smoke),
                round(100 * smoke.str.contains(SMOKING_NEVER).mean(), 1), n_total)]

    alcohol = life22[ALCOHOL_COL].dropna().astype(str).str[0].astype(int)
    for label, mask in [("Lifetime abstainer", alcohol == 1),
                        ("Non- or rare-drinker (< monthly)", alcohol <= 2),
                        ("Regular drinker (weekly or more)", alcohol >= 5)]:
        rows.append(row("Alcohol (2022)", label, len(alcohol), round(100 * mask.mean(), 1), n_total))

    days = []
    for year in (22, 23, 24):
        life = read_sheet(year, "건강생활").reindex(rc_ids).dropna(how="all")
        col = next(c for c in life.columns if any(k in c for k in EXERCISE_KEYS))
        d = pd.to_numeric(life[col], errors="coerce").dropna()
        days.append(pd.DataFrame({"year": year, "days": d}))
    days = pd.concat(days)
    daily = days.groupby("year")["days"].apply(lambda d: 100 * (d == 7).mean())
    rows.append(row("Physical activity (2022-2024)", "Exercise, mean days/week", len(days),
                    round(days["days"].mean(), 2), n_total))
    rows.append(row("Physical activity (2022-2024)", "Exercise daily, by wave (%)", len(days),
                    "; ".join(f"20{y}: {v:.1f}" for y, v in daily.items()), n_total))
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    meta, _, _ = load_all_data()
    rc_ids = meta.index[meta["cohort"] == "Muan"]
    n_total = len(rc_ids)

    table = pd.DataFrame(diet_rows(n_total) + lifestyle_rows(rc_ids, n_total))
    table.to_csv(OUT / "table3_lifestyle.csv", index=False)
    for _, r in table.iterrows():
        print(f"  {r['characteristic']}: n = {r['n']}, {r['value']} ({r['completion_pct']}%)")
    print(f"  Saved to {OUT}")


if __name__ == "__main__":
    main()

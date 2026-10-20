"""
Longevity Village Analysis: Table S4 (diet-microbiome correlations).

Spearman correlations between 16 food-group frequencies from the 2025 food
frequency questionnaire (codes 1-8) and the CLR abundance of 11 target genera,
in RC participants of the 2025 wave with matched microbiome data (n = 145;
pairwise-complete n per test). BH-FDR across all 176 tests.

The output tableS4_diet_microbiome.csv is also the input of the Fig. S4
heatmap in longevity_extra.py (fig_diet_microbiome).

Usage (set paths in ml/longevity_config.py first):
    cd /path/to/kme-microbiome-analysis
    python -m ml.longevity_table_s4
"""
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from ml.data_join import clr_transform
from ml.longevity_config import LONGEVITY_OUT
from ml.longevity_muan import load_all_data
from ml.longevity_table3 import read_sheet

OUT = LONGEVITY_OUT

# Food group -> 2025 FFQ "overall frequency" column prefix
FOODS = {
    "Meat": "육류", "Fish": "생선", "Seafood": "해산물", "Seaweed": "해조류",
    "Vegetables": "채소", "Namul": "나물", "Mushrooms": "버섯", "Kimchi": "김치",
    "Jeotgal": "장아찌/젓갈", "Other veg": "기타채소", "Fruits": "생과일", "Snacks": "간식",
    "Beverages": "음료", "Nuts": "견과류", "Eggs": "달걀 요리", "Dairy": "유제품",
}
GENERA = [
    "Lactobacillus", "Limosilactobacillus", "Ligilactobacillus", "Megasphaera", "Olsenella",
    "Escherichia-Shigella", "Methanobrevibacter", "Bifidobacterium", "Roseburia", "Coprococcus",
    "[Eubacterium] eligens group",
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    meta, genus_rel, _ = load_all_data()
    genus_clr = clr_transform(genus_rel)

    ffq = read_sheet(25, "식생")
    rc25 = [s for s in meta.index[meta["cohort"] == "Muan"] if s.startswith("25_") and s in ffq.index]

    rows = []
    for food, key in FOODS.items():
        freq = pd.to_numeric(ffq.loc[rc25, f"{key} 전체 섭취 횟수"], errors="coerce").dropna()
        for genus in GENERA:
            rho, p = stats.spearmanr(freq, genus_clr.loc[freq.index, genus])
            rows.append({"food": food, "genus": genus, "rho": rho, "p": p, "n": len(freq)})
    df = pd.DataFrame(rows)
    _, df["q"], _, _ = multipletests(df["p"], method="fdr_bh")
    df.to_csv(OUT / "tableS4_diet_microbiome.csv", index=False)

    print(f"  {len(rc25)} RC participants with 2025 FFQ and microbiome data; "
          f"{len(df)} tests, pairwise n = {df['n'].min()}-{df['n'].max()}")
    print(f"  uncorrected P < 0.05: {(df['p'] < 0.05).sum()}; FDR q < 0.05: {(df['q'] < 0.05).sum()}")
    for r in df.sort_values("p").head(3).itertuples():
        print(f"    {r.food} x {r.genus}: rho = {r.rho:+.2f}, P = {r.p:.3f}, q = {r.q:.2f}")
    print(f"  Saved to {OUT}")


if __name__ == "__main__":
    main()

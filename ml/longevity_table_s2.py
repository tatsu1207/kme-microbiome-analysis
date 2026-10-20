"""
Longevity Village Analysis: Table S2 (residential distribution of the DS cohort).

DS participants with microbiome data are matched to hospital registration
addresses by screening number (2020: the original number, falling back to
the re-issued number). Addresses are classified with explicit rules:

  Gwangju metropolitan city
  Naju-si:  RC sub-districts (Nampyeong-eup, Gongsan-, Seji-, Bonghwang-,
            Noan-, Dasi-, Donggang-myeon, Geumnam-dong incl. 금남길);
            Bitgaram new town (road names listed in BITGARAM_ROADS);
            other dong; rural non-RC myeon; apartment/villa without dong;
            unclassified (no sub-district information)
  Other RC sub-districts outside Naju (Imja-myeon, Sinan; Moksadong-myeon, Gokseong)
  Other Jeollanam-do cities/counties; other provinces

Only aggregate counts are written; addresses never leave the input file.

Usage (set paths in ml/longevity_config.py first):
    cd /path/to/kme-microbiome-analysis
    python -m ml.longevity_table_s2
"""
import re

import pandas as pd

from ml.longevity_config import DS_ADDRESSES, LONGEVITY_ASV_TABLE, LONGEVITY_OUT

OUT = LONGEVITY_OUT

# RC sub-districts: address keyword(s) -> label
NAJU_RC = {
    "남평읍": ("남평읍",), "공산면": ("공산면",), "세지면": ("세지면",), "봉황면": ("봉황면",),
    "노안면": ("노안면",), "다시면": ("다시면",), "동강면": ("동강면",), "금남동": ("금남동", "금남길"),
}
OTHER_RC = {"임자면": ("Sinan-gun", "임자"), "목사동면": ("Gokseong-gun", "목사동")}
# Roads of the Bitgaram innovation-city district (빛가람동)
BITGARAM_ROADS = ("빛가람", "그린로", "한빛로", "우정로", "중야", "상야", "호수로", "전력로",
                  "문화로", "교육길", "배멧", "세계로", "빛누리", "혁신")
JEONNAM = ("전남", "전라남도", "목포", "여수", "순천", "광양", "담양", "곡성", "구례", "고흥",
           "보성", "화순", "장흥", "강진", "해남", "영암", "무안", "함평", "영광", "장성", "완도",
           "진도", "신안")
OTHER_PROVINCES = ("서울", "경기", "인천", "강원", "충북", "충청", "충남", "대전", "세종", "전북",
                   "전라북도", "전주", "경북", "경남", "경상", "대구", "부산", "울산", "제주",
                   "김포", "창원")
APARTMENT = re.compile(r"아파트|빌라|APT|타운|맨션|파크|힐스|하우스")


def normalize_id(raw):
    """20-DS-36 / 20_DS_0036 -> 20_DS_0036"""
    parts = str(raw).strip().replace("-", "_").split("_")
    try:
        return f"{parts[0]}_{parts[1]}_{int(parts[2]):04d}"
    except (IndexError, ValueError):
        return None


def load_addresses(ds_ids):
    """DS sample ID -> address, original screening number first."""
    original, reissued = {}, {}
    for sheet in pd.ExcelFile(DS_ADDRESSES).sheet_names:
        df = pd.read_excel(DS_ADDRESSES, sheet_name=sheet)
        id_col = next(c for c in df.columns if "스크리닝" in str(c) and "바뀐" not in str(c))
        for col, store in [(id_col, original), ("바뀐 스크리닝 번호", reissued)]:
            if col not in df.columns:
                continue
            for raw, address in zip(df[col], df["주소"]):
                key = normalize_id(raw)
                if key and pd.notna(address) and str(address).strip():
                    store.setdefault(key, str(address))
    return {s: original.get(s, reissued.get(s)) for s in ds_ids if s in original or s in reissued}


def classify(address):
    a = re.sub(r"\s+", " ", address).strip()
    if a.startswith(OTHER_PROVINCES):
        return "Other provinces", "—"
    if "광주" in a:
        return "Gwangju metropolitan", "Urban"
    if "나주" in a or "동신대" in a:
        for label, keys in NAJU_RC.items():
            if any(k in a for k in keys):
                kind = "Urban" if label.endswith("동") else "Rural"
                return "Naju-si", f"{kind}, RC area ({label})"
        if any(r in a for r in BITGARAM_ROADS):
            return "Naju-si", "Urban (Bitgaram district)"
        if re.search(r"\S+면(\s|$)", a):
            return "Naju-si", "Rural, non-RC area (면)"
        if re.search(r"\S+동(\s|$|\d)", a) or "동신대" in a:
            return "Naju-si", "Urban (other dong)"
        if APARTMENT.search(a):
            return "Naju-si", "Urban (apartment/villa)"
        return "Naju-si", "Unclassified"
    for label, (county, key) in OTHER_RC.items():
        if key in a:
            return county, f"Rural, RC area ({label})"
    if any(j in a for j in JEONNAM):
        return "Jeonnam (other)", "—"
    return "Unparsed", "—"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    samples = pd.read_csv(LONGEVITY_ASV_TABLE, sep="\t", index_col=0, nrows=0).columns
    ds_ids = [s for s in samples if "_DS_" in s]
    addresses = load_addresses(ds_ids)

    cats = pd.Series({s: classify(a) for s, a in addresses.items()})
    table = (cats.value_counts().rename_axis("key").reset_index(name="n"))
    table[["region", "category"]] = pd.DataFrame(table["key"].tolist(), index=table.index)
    table["pct"] = (100 * table["n"] / len(cats)).round(1)
    table = table[["region", "category", "n", "pct"]].sort_values("n", ascending=False)
    table.to_csv(OUT / "tableS2_ds_residence.csv", index=False)

    rc_area = table[table["category"].str.match(r"(Rural|Urban), RC area")]["n"].sum()
    urban_naju = table[(table["region"] == "Naju-si") & table["category"].str.startswith("Urban (")]["n"].sum()
    gwangju = table[table["region"] == "Gwangju metropolitan"]["n"].sum()
    n = len(cats)
    print(f"  Matched {n} of {len(ds_ids)} DS samples")
    print(table.to_string(index=False))
    print(f"  Gwangju {gwangju} ({100 * gwangju / n:.1f}%); urban Naju excl. RC sites {urban_naju} "
          f"({100 * urban_naju / n:.1f}%); RC sub-districts {rc_area} ({100 * rc_area / n:.1f}%)")
    print(f"  Saved to {OUT}")


if __name__ == "__main__":
    main()

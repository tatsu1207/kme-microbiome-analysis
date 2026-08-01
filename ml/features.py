"""
Step 2: Feature engineering.

Processes raw questionnaire columns into ML-ready numeric features.
Assembles feature matrices for three model tiers:
  M1: questionnaire only
  M2: microbiome only (genus + pathways)
  M3: combined
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ml.config import QUESTIONNAIRE_BLOCKS


# ---------------------------------------------------------------------------
# Questionnaire preprocessing
# ---------------------------------------------------------------------------
def _is_numeric_like(series: pd.Series, threshold: float = 0.8) -> bool:
    """Check if a string series is mostly numeric."""
    cleaned = series.replace("", np.nan).dropna()
    if len(cleaned) == 0:
        return False
    numeric_count = pd.to_numeric(cleaned, errors="coerce").notna().sum()
    return (numeric_count / len(cleaned)) >= threshold


def _encode_sex(series: pd.Series) -> pd.Series:
    return series.map({"M": 0, "F": 1}).astype(float)


def process_questionnaire(quest: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """
    Convert questionnaire string columns to numeric features.
    - Numeric-like columns → float + standard scaling
    - Categorical columns with <=10 unique → one-hot encoding
    - High-cardinality or free-text columns → dropped
    """
    df = quest.copy()

    # Separate numeric vs categorical
    numeric_cols = []
    categorical_cols = []
    drop_cols = []

    for col in df.columns:
        # Skip CD (code) columns that are just normal-range flags
        if col.endswith("CD") or col.endswith("CDYN") or col.endswith("YN"):
            # Keep boolean Y/N columns
            uniq = df[col].replace("", np.nan).dropna().unique()
            if set(uniq) <= {"Y", "N", "1", "2"}:
                categorical_cols.append(col)
            else:
                drop_cols.append(col)
            continue

        if _is_numeric_like(df[col]):
            numeric_cols.append(col)
        else:
            nuniq = df[col].replace("", np.nan).dropna().nunique()
            if 2 <= nuniq <= 10:
                categorical_cols.append(col)
            elif nuniq <= 1:
                drop_cols.append(col)  # constant
            else:
                drop_cols.append(col)  # high cardinality / free text

    # Convert numeric columns
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # One-hot encode categoricals
    cat_dummies = []
    for col in categorical_cols:
        series = df[col].replace("", np.nan)
        dummies = pd.get_dummies(series, prefix=col, dummy_na=False, dtype=float)
        # Drop first to avoid multicollinearity (if >2 categories)
        if dummies.shape[1] > 2:
            dummies = dummies.iloc[:, 1:]
        cat_dummies.append(dummies)

    # Assemble
    result = df[numeric_cols].copy()
    if cat_dummies:
        result = pd.concat([result] + cat_dummies, axis=1)

    return result


# ---------------------------------------------------------------------------
# Assign columns to questionnaire blocks
# ---------------------------------------------------------------------------
def assign_blocks(quest_processed: pd.DataFrame) -> dict[str, list[str]]:
    """Map processed questionnaire columns to their conceptual blocks."""
    all_cols = list(quest_processed.columns)
    block_assignments = {}

    for block_name, spec in QUESTIONNAIRE_BLOCKS.items():
        matched = []
        # Match by exact column names
        if "cols" in spec:
            for c in spec["cols"]:
                matched.extend([col for col in all_cols if col == c or col.startswith(c + "_")])
        # Match by prefix
        if "prefix" in spec:
            for prefix in spec["prefix"]:
                matched.extend([col for col in all_cols if col.startswith(prefix)])
        block_assignments[block_name] = sorted(set(matched))

    # Unassigned columns go to "other"
    assigned = set()
    for cols in block_assignments.values():
        assigned.update(cols)
    block_assignments["other"] = [c for c in all_cols if c not in assigned]

    return block_assignments


# ---------------------------------------------------------------------------
# Assemble feature matrices for M1, M2, M3
# ---------------------------------------------------------------------------
def build_feature_matrices(
    quest_processed: pd.DataFrame,
    genus_clr: pd.DataFrame,
    pathways: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """
    Returns dict with feature matrices:
      M1: questionnaire only
      M2: microbiome only (genus CLR + pathway CLR)
      M3: M1 + M2 combined
    All share the same row index.
    """
    # Align indices
    common_idx = quest_processed.index.intersection(genus_clr.index)
    if len(pathways.columns) > 0:
        common_idx = common_idx.intersection(pathways.index)
    q = quest_processed.loc[common_idx]
    g = genus_clr.loc[common_idx]

    m1 = q
    if len(pathways.columns) > 0:
        p = pathways.loc[common_idx]
        m2 = pd.concat([g, p], axis=1)
        m3 = pd.concat([q, g, p], axis=1)
    else:
        m2 = g
        m3 = pd.concat([q, g], axis=1)

    return {"M1": m1, "M2": m2, "M3": m3, "index": common_idx}


def print_feature_summary(
    feature_matrices: dict[str, pd.DataFrame],
    block_assignments: dict[str, list[str]],
) -> None:
    print("\n=== Feature Matrix Summary ===")
    for name in ["M1", "M2", "M3"]:
        mat = feature_matrices[name]
        print(f"  {name}: {mat.shape[0]} samples × {mat.shape[1]} features")

    print("\n=== Questionnaire Blocks ===")
    for block, cols in block_assignments.items():
        print(f"  {block}: {len(cols)} features")

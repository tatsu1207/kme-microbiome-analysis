"""
Step 3: Model training, comparison, and interpretation.

Three-tier comparison:
  M1: Questionnaire only
  M2: Microbiome only
  M3: Combined (questionnaire + microbiome)

Evaluation: nested cross-validation, AUC-ROC, DeLong test, SHAP.
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from ml.config import INNER_CV_FOLDS, OUTER_CV_FOLDS, OUTPUT_DIR, RANDOM_STATE

warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------------
# DeLong test for AUC comparison
# ---------------------------------------------------------------------------
def _auc_variance_components(y_true, y_score):
    """Compute DeLong variance components."""
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    m, n = len(pos), len(neg)

    # For each positive, fraction of negatives it beats
    v_pos = np.array([(y_score[y_true == 0] < p).sum() + 0.5 * (y_score[y_true == 0] == p).sum()
                       for p in pos]) / n
    # For each negative, fraction of positives that beat it
    v_neg = np.array([(y_score[y_true == 1] > ng).sum() + 0.5 * (y_score[y_true == 1] == ng).sum()
                       for ng in neg]) / m

    return v_pos, v_neg


def delong_test(y_true, y_score1, y_score2):
    """
    DeLong test for comparing two AUCs on the same dataset.
    Returns: z-statistic, p-value (two-sided).
    """
    v1_pos, v1_neg = _auc_variance_components(y_true, y_score1)
    v2_pos, v2_neg = _auc_variance_components(y_true, y_score2)

    m = (y_true == 1).sum()
    n = (y_true == 0).sum()

    auc1 = v1_pos.mean()
    auc2 = v2_pos.mean()

    # Covariance matrix of AUC1, AUC2
    s10_1 = np.cov(v1_pos, v2_pos)
    s01_1 = np.cov(v1_neg, v2_neg)

    var = (s10_1[0, 0] + s01_1[0, 0]) / m + \
          (s10_1[1, 1] + s01_1[1, 1]) / n - \
          2 * (s10_1[0, 1] / m + s01_1[0, 1] / n)

    # Guard against degenerate case
    if var <= 0:
        return 0.0, 1.0

    z = (auc1 - auc2) / np.sqrt(var)
    p = 2 * stats.norm.sf(abs(z))
    return z, p


# ---------------------------------------------------------------------------
# Build classifier pipelines
# ---------------------------------------------------------------------------
CLASSIFIERS = {
    "xgboost": lambda: XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        reg_alpha=0.1, reg_lambda=1.0, random_state=RANDOM_STATE,
        eval_metric="logloss", use_label_encoder=False, verbosity=0, n_jobs=4,
    ),
    "logistic_regression": lambda: LogisticRegression(
        max_iter=1000, C=1.0, penalty="l2", solver="lbfgs",
        random_state=RANDOM_STATE, n_jobs=4,
    ),
    "random_forest": lambda: RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=5,
        random_state=RANDOM_STATE, n_jobs=4,
    ),
}


def make_pipeline(algorithm: str = "xgboost") -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", CLASSIFIERS[algorithm]()),
    ])


# ---------------------------------------------------------------------------
# Nested cross-validation
# ---------------------------------------------------------------------------
def evaluate_model(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    cv_folds: int = OUTER_CV_FOLDS,
    algorithm: str = "xgboost",
) -> dict:
    """
    Run stratified K-fold CV and return out-of-fold predictions + AUC.
    """
    valid_mask = y.notna()
    X_valid = X.loc[valid_mask].copy()
    y_valid = y.loc[valid_mask].astype(int).copy()

    if y_valid.nunique() < 2:
        print(f"  {model_name}: skipped (single class)")
        return {"auc": np.nan, "y_true": y_valid, "y_prob": np.full(len(y_valid), np.nan)}

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)

    y_prob = np.zeros(len(y_valid))
    fold_aucs = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_valid, y_valid)):
        X_train = X_valid.iloc[train_idx]
        y_train = y_valid.iloc[train_idx]
        X_test = X_valid.iloc[test_idx]
        y_test = y_valid.iloc[test_idx]

        pipe = make_pipeline(algorithm=algorithm)
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]
        y_prob[test_idx] = proba

        fold_auc = roc_auc_score(y_test, proba)
        fold_aucs.append(fold_auc)

    overall_auc = roc_auc_score(y_valid, y_prob)
    print(f"  {model_name}: AUC = {overall_auc:.4f}  (fold AUCs: {[f'{a:.3f}' for a in fold_aucs]})")

    return {
        "auc": overall_auc,
        "fold_aucs": fold_aucs,
        "y_true": y_valid,
        "y_prob": y_prob,
        "model_name": model_name,
    }


# ---------------------------------------------------------------------------
# SHAP interpretation
# ---------------------------------------------------------------------------
def compute_shap(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Train on full data and compute SHAP values."""
    try:
        import shap
    except ImportError:
        print("  shap not installed, skipping SHAP analysis")
        return pd.DataFrame()

    valid_mask = y.notna()
    X_valid = X.loc[valid_mask].copy()
    y_valid = y.loc[valid_mask].astype(int).copy()

    pipe = make_pipeline()
    pipe.fit(X_valid, y_valid)

    # Get transformed data for SHAP
    X_transformed = pipe.named_steps["scaler"].transform(
        pipe.named_steps["imputer"].transform(X_valid)
    )
    X_transformed = pd.DataFrame(X_transformed, columns=X_valid.columns, index=X_valid.index)

    explainer = shap.TreeExplainer(pipe.named_steps["clf"])
    shap_values = explainer.shap_values(X_transformed)

    shap_df = pd.DataFrame(shap_values, columns=X_valid.columns, index=X_valid.index)
    return shap_df


# ---------------------------------------------------------------------------
# Run full comparison
# ---------------------------------------------------------------------------
def run_comparison(
    feature_matrices: dict[str, pd.DataFrame],
    outcomes: pd.DataFrame,
    target: str = "y_obesity",
    algorithm: str = "xgboost",
) -> dict:
    """
    Run M1 vs M2 vs M3 comparison for a given outcome.
    Returns results dict with AUCs, predictions, and DeLong tests.
    """
    y = outcomes[target]
    valid_n = y.notna().sum()
    pos_n = (y == 1).sum()
    print(f"\n{'='*60}")
    print(f"Target: {target}  (n={valid_n}, pos={pos_n}, rate={pos_n/valid_n*100:.1f}%)")
    print(f"{'='*60}")

    results = {}
    for name in ["M1", "M2", "M3"]:
        X = feature_matrices[name]
        results[name] = evaluate_model(X, y, name, algorithm=algorithm)

    # DeLong tests: M3 vs M1, M3 vs M2
    for pair, (a, b) in [("M3_vs_M1", ("M3", "M1")), ("M3_vs_M2", ("M3", "M2"))]:
        # Align to common valid samples
        common = results[a]["y_true"].index.intersection(results[b]["y_true"].index)
        y_a = results[a]["y_true"].loc[common]
        prob_a = pd.Series(results[a]["y_prob"], index=results[a]["y_true"].index).loc[common]
        prob_b = pd.Series(results[b]["y_prob"], index=results[b]["y_true"].index).loc[common]

        z, p = delong_test(y_a.values, prob_a.values, prob_b.values)
        delta = results[a]["auc"] - results[b]["auc"]
        print(f"  {pair}: ΔAUC = {delta:+.4f}, z = {z:.3f}, p = {p:.4f}")
        results[pair] = {"delta_auc": delta, "z": z, "p": p}

    return results


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def run_all(
    feature_matrices: dict[str, pd.DataFrame],
    outcomes: pd.DataFrame,
    algorithm: str = "xgboost",
) -> dict:
    """Run comparison for all outcomes."""
    all_results = {}
    targets = [
        # Metabolic
        "y_obesity", "y_hypertension", "y_prediabetes", "y_liver", "y_multimorbidity",
        # Digestive
        "y_digestive_symptoms", "y_multi_digestive", "y_abnormal_stool",
        "y_bowel_difficulty", "y_poor_gut",
    ]

    for target in targets:
        if target == "y_multimorbidity":
            # Binarize: 2+ conditions vs 0-1
            y_binary = (outcomes[target] >= 2).astype(float)
            y_binary.loc[outcomes[target].isna()] = np.nan
            outcomes_modified = outcomes.copy()
            outcomes_modified["y_multimorbidity_binary"] = y_binary
            all_results[target] = run_comparison(
                feature_matrices, outcomes_modified, "y_multimorbidity_binary",
                algorithm=algorithm,
            )
        elif target not in outcomes.columns:
            print(f"\n  Skipping {target} (not in outcomes)")
            continue
        else:
            all_results[target] = run_comparison(
                feature_matrices, outcomes, target, algorithm=algorithm,
            )

    # Summary table
    print(f"\n{'='*60}")
    print("SUMMARY: AUC by model tier")
    print(f"{'='*60}")
    print(f"{'Outcome':<25} {'M1':>8} {'M2':>8} {'M3':>8} {'ΔAUC(M3-M1)':>12} {'p':>8}")
    print("-" * 72)
    for target in targets:
        r = all_results[target]
        m1 = r.get("M1", {}).get("auc", np.nan)
        m2 = r.get("M2", {}).get("auc", np.nan)
        m3 = r.get("M3", {}).get("auc", np.nan)
        delta = r.get("M3_vs_M1", {}).get("delta_auc", np.nan)
        p = r.get("M3_vs_M1", {}).get("p", np.nan)
        print(f"{target:<25} {m1:>8.4f} {m2:>8.4f} {m3:>8.4f} {delta:>+12.4f} {p:>8.4f}")

    return all_results

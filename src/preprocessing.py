"""
Data Preprocessing Pipeline — Kaggle Credit Card Fraud Dataset
===============================================================
Handles feature engineering, scaling, and class imbalance via SMOTE.

Dataset specifics:
  - V1–V28: Already PCA-transformed and standardized by the data provider.
             We do NOT re-scale these — they are already zero-mean, unit-variance.
  - Time:   Seconds since first transaction. Encodes time-of-day fraud patterns.
             We apply CYCLIC encoding (sin/cos) after converting to hour-of-day.
  - Amount: Raw EUR amount. Right-skewed → log1p transform applied.
  - Class:  Target (0=legit, 1=fraud). 0.173% fraud rate.

Design Decisions:
  - SMOTE applied ONLY on training set (critical: never on test/val)
  - V1–V28 passed through as-is (already standardized by PCA pipeline)
  - Only Amount_log and Time features are scaled via StandardScaler
  - StandardScaler fitted on train only, applied to test (no leakage)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib
import os
import sys

# V1–V28 pass through as-is (pre-standardized by PCA)
V_COLS = [f"V{i}" for i in range(1, 29)]
FEATURE_COLS = V_COLS + ["Amount_log", "Time_sin", "Time_cos"]
TARGET_COL = "Class"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply domain-driven feature transformations to the Kaggle CC fraud dataset.

    Transformations:
    - Amount_log: log1p of Amount (handles extreme right skew; Amount=0 → safe)
    - Time_sin / Time_cos: Cyclic encoding of hour-of-day derived from Time.
        Time is cumulative seconds over ~48h. We extract hour-of-day via
        (Time % 86400) / 3600, then encode cyclically so 23:00→00:00 is
        treated as adjacent (not maximally distant).
    - V1–V28: Passed through unchanged (already PCA-transformed & standardized)
    - Original Time and Amount columns dropped post-transformation.

    Args:
        df: Raw Kaggle creditcard DataFrame

    Returns:
        DataFrame with engineered features
    """
    df = df.copy()

    # log1p transform: handles Amount=0 safely, reduces skew impact
    df["Amount_log"] = np.log1p(df["Amount"])

    # Cyclic time-of-day encoding
    seconds_in_day = 86400
    hour_of_day = (df["Time"] % seconds_in_day) / 3600  # float in [0, 24)
    df["Time_sin"] = np.sin(2 * np.pi * hour_of_day / 24)
    df["Time_cos"] = np.cos(2 * np.pi * hour_of_day / 24)

    df.drop(columns=["Amount", "Time"], inplace=True, errors="ignore")
    return df


def preprocess(
    df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.1,
    apply_smote: bool = True,
    smote_ratio: float = 0.1,
    random_state: int = 42,
    scaler_path: str = None,
) -> dict:
    """
    Full preprocessing pipeline: engineer → split → scale → balance.

    SMOTE ratio note:
        At 0.173% fraud rate, we target 10% minority ratio (smote_ratio=0.1).
        More conservative than typical — at extreme imbalance, aggressive
        oversampling creates synthetic samples far from the true fraud manifold.
        Training set has ~394 fraud samples; SMOTE generates ~22,000 to reach 10%.

    Args:
        df: Raw Kaggle DataFrame (284,807 rows)
        test_size: Fraction for test set (default 0.2 → ~56,961 rows)
        val_size: Fraction for validation set (default 0.1 → ~28,480 rows)
        apply_smote: Whether to apply SMOTE oversampling
        smote_ratio: Target minority:majority ratio after SMOTE
        random_state: Reproducibility seed
        scaler_path: If provided, saves fitted scaler to this path

    Returns:
        dict with keys: X_train, X_val, X_test, y_train, y_val, y_test,
                        feature_names, class_counts, scaler
    """
    df = engineer_features(df)

    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values

    # Stratified split — critical at 0.17% minority class
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    val_fraction = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_fraction, stratify=y_temp,
        random_state=random_state
    )

    # Scale features — fit ONLY on train
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    if scaler_path:
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        joblib.dump(scaler, scaler_path)
        print(f"[Preprocessing] Scaler saved to {scaler_path}")

    counts_before = {
        int(k): int(v)
        for k, v in zip(*np.unique(y_train, return_counts=True))
    }

    if apply_smote:
        smote = SMOTE(
            sampling_strategy=smote_ratio,
            random_state=random_state,
            k_neighbors=5
        )
        X_train, y_train = smote.fit_resample(X_train, y_train)

    counts_after = {
        int(k): int(v)
        for k, v in zip(*np.unique(y_train, return_counts=True))
    }

    print(f"[Preprocessing] Split → Train: {len(X_train):,} | "
          f"Val: {len(X_val):,} | Test: {len(X_test):,}")
    print(f"[Preprocessing] Before SMOTE: "
          f"legit={counts_before[0]:,}, fraud={counts_before[1]:,} "
          f"({counts_before[1]/sum(counts_before.values()):.3%})")
    if apply_smote:
        print(f"[Preprocessing] After SMOTE:  "
              f"legit={counts_after[0]:,}, fraud={counts_after[1]:,} "
              f"({counts_after[1]/sum(counts_after.values()):.3%})")

    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "feature_names": FEATURE_COLS,
        "class_counts": counts_after,
        "scaler": scaler,
    }


def load_and_preprocess(scaler_path: str = None, **kwargs) -> dict:
    """
    Convenience wrapper: auto-locate dataset and run full preprocessing.

    Args:
        scaler_path: Optional path to save fitted scaler
        **kwargs: Passed to preprocess()

    Returns:
        Preprocessing result dict
    """
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data.load_kaggle_data import get_dataset

    df = get_dataset()
    print(f"[Preprocessing] Fraud rate: {df['Class'].mean():.4%} "
          f"({int(df['Class'].sum())} / {len(df):,})")
    return preprocess(df, scaler_path=scaler_path, **kwargs)
"""
Kaggle Credit Card Fraud Dataset Loader
========================================
Dataset: Credit Card Fraud Detection — MLG-ULB (Université Libre de Bruxelles)
Source:  https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Schema (31 columns):
  Time   — Seconds elapsed since first transaction in dataset (~48h window)
  V1–V28 — PCA-transformed features (original features anonymized for privacy)
  Amount — Transaction amount in EUR
  Class  — 0 = legitimate, 1 = fraudulent

Dataset stats:
  284,807 total transactions
  492 fraudulent (0.173% fraud rate)
  0 missing values
  2 days of European cardholder transactions
"""

import pandas as pd
import numpy as np
import os
import sys

EXPECTED_ROWS = 284807
EXPECTED_FRAUD = 492
EXPECTED_COLS = 31
DATA_DIR = os.path.dirname(os.path.abspath(__file__))


def load_and_validate(path: str) -> pd.DataFrame:
    """
    Load the Kaggle creditcard dataset and validate its integrity.
    Accepts .csv or .csv.gz formats.
    """
    print(f"[Loader] Reading: {path}")

    if path.endswith(".gz"):
        df = pd.read_csv(path, compression="gzip")
    else:
        df = pd.read_csv(path)

    # --- Schema validation ---
    expected_cols = ["Time", "Amount", "Class"] + [f"V{i}" for i in range(1, 29)]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    # --- Null check ---
    nulls = df.isnull().sum().sum()
    if nulls > 0:
        print(f"[Loader] WARNING: {nulls} null values found. Will drop.")
        df = df.dropna()

    # --- Stats report ---
    n_fraud = int(df["Class"].sum())
    n_legit = len(df) - n_fraud
    fraud_rate = df["Class"].mean()

    print(f"[Loader] ✓ Loaded {len(df):,} transactions")
    print(f"[Loader]   Legitimate: {n_legit:,} ({1 - fraud_rate:.4%})")
    print(f"[Loader]   Fraudulent: {n_fraud:,} ({fraud_rate:.4%})")
    print(f"[Loader]   Time range: {df['Time'].min():.0f}s – {df['Time'].max():.0f}s "
          f"(~{df['Time'].max() / 3600:.1f} hours)")
    print(f"[Loader]   Amount range: €{df['Amount'].min():.2f} – €{df['Amount'].max():,.2f} "
          f"(median: €{df['Amount'].median():.2f})")

    if len(df) != EXPECTED_ROWS:
        print(f"[Loader] WARNING: Expected {EXPECTED_ROWS:,} rows, got {len(df):,}")
    if n_fraud != EXPECTED_FRAUD:
        print(f"[Loader] WARNING: Expected {EXPECTED_FRAUD} fraud cases, got {n_fraud}")

    return df


def get_dataset(gz_path: str = None, csv_path: str = None) -> pd.DataFrame:
    """
    Convenience function: load from .gz or .csv, whichever exists.

    Priority:
    1. Explicit gz_path argument
    2. Explicit csv_path argument
    3. Auto-detect in data/ directory
    """
    # Priority 1: explicit gz
    if gz_path and os.path.exists(gz_path):
        return load_and_validate(gz_path)

    # Priority 2: explicit csv
    if csv_path and os.path.exists(csv_path):
        return load_and_validate(csv_path)

    # Priority 3: auto-detect in data/
    auto_gz = os.path.join(DATA_DIR, "creditcard.csv.gz")
    auto_csv = os.path.join(DATA_DIR, "creditcard.csv")

    if os.path.exists(auto_gz):
        return load_and_validate(auto_gz)
    if os.path.exists(auto_csv):
        return load_and_validate(auto_csv)

    raise FileNotFoundError(
        "Dataset not found. Download from:\n"
        "  https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud\n"
        f"Place creditcard.csv or creditcard.csv.gz in: {DATA_DIR}"
    )


if __name__ == "__main__":
    df = get_dataset()
    print("\n[Loader] Sample rows:")
    print(df[["Time", "Amount", "Class"]].head(5).to_string())
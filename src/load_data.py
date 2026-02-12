"""Utility module for loading and preparing the Wingspan bird card dataset."""

import json
from pathlib import Path

import pandas as pd

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
DATA_FILE = ASSETS_DIR / "master_bird_data.json"


def load_bird_data() -> pd.DataFrame:
    """Load master_bird_data.json into a pandas DataFrame.

    Returns:
        DataFrame with one row per bird card.
    """
    with open(DATA_FILE, encoding="utf-8") as f:
        raw = json.load(f)
    return pd.DataFrame(raw)


if __name__ == "__main__":
    df = load_bird_data()
    print(f"Loaded {len(df)} bird cards with {len(df.columns)} columns.\n")
    print("Column names:")
    for col in df.columns:
        non_null = df[col].notna().sum()
        print(f"  {col:<30} {non_null:>4} non-null  ({df[col].dtype})")
    print(f"\nShape: {df.shape}")

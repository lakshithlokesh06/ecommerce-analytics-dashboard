"""
data_loader.py
--------------
Data ingestion, cleaning, and validation pipeline.
"""

import pandas as pd
import numpy as np


def load_data(file) -> pd.DataFrame:
    """
    Load CSV data from a file path or file-like object (Streamlit uploader).
    Returns a raw DataFrame.
    """
    df = pd.read_csv(file)
    return df


def validate_columns(df: pd.DataFrame) -> list[str]:
    """
    Check that the required columns exist.  Returns a list of missing column
    names (empty list if all present).
    """
    required = {
        "order_id", "order_date", "product_name", "category",
        "quantity", "unit_price", "revenue", "region",
    }
    missing = sorted(required - set(df.columns))
    return missing


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full cleaning pipeline:
      1. Parse dates
      2. Impute missing values
      3. Enforce types
      4. Remove duplicates
      5. Derive helper columns
    """
    df = df.copy()

    # 1. Parse dates
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    df = df.dropna(subset=["order_date"])

    # 2. Impute missing values
    if "customer_name" in df.columns:
        df["customer_name"] = df["customer_name"].fillna("Unknown")
    if "city" in df.columns:
        df["city"] = df["city"].fillna("Unknown")
    if "discount" in df.columns:
        df["discount"] = pd.to_numeric(df["discount"], errors="coerce").fillna(0.0)
    if "shipping_cost" in df.columns:
        df["shipping_cost"] = pd.to_numeric(df["shipping_cost"], errors="coerce").fillna(
            df["shipping_cost"].median() if df["shipping_cost"].notna().any() else 0.0
        )

    # 3. Enforce numeric types
    for col in ["quantity", "unit_price", "revenue"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # 4. Remove exact duplicates
    df = df.drop_duplicates(subset=["order_id"], keep="first")

    # 5. Derive helper columns
    df["year"] = df["order_date"].dt.year
    df["month"] = df["order_date"].dt.month
    df["month_name"] = df["order_date"].dt.strftime("%b")
    df["quarter"] = df["order_date"].dt.quarter
    df["day_of_week"] = df["order_date"].dt.day_name()
    df["year_month"] = df["order_date"].dt.to_period("M").astype(str)

    # Recalculate revenue if missing
    mask = df["revenue"] == 0
    if "discount" in df.columns:
        df.loc[mask, "revenue"] = (
            df.loc[mask, "quantity"]
            * df.loc[mask, "unit_price"]
            * (1 - df.loc[mask, "discount"])
        ).round(2)
    else:
        df.loc[mask, "revenue"] = (
            df.loc[mask, "quantity"] * df.loc[mask, "unit_price"]
        ).round(2)

    df = df.sort_values("order_date").reset_index(drop=True)
    return df


def get_cleaning_summary(raw_df: pd.DataFrame, clean_df: pd.DataFrame) -> dict:
    """Return a summary of what the cleaning step changed."""
    return {
        "rows_before": len(raw_df),
        "rows_after": len(clean_df),
        "duplicates_removed": len(raw_df) - len(clean_df),
        "missing_before": int(raw_df.isnull().sum().sum()),
        "missing_after": int(clean_df.isnull().sum().sum()),
        "columns": list(clean_df.columns),
    }

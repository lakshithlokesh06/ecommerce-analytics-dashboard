"""
utils.py
--------
Utility / helper functions.
"""

import pandas as pd
import io


def format_currency(value: float) -> str:
    """Format a number as USD currency string."""
    if abs(value) >= 1_000_000:
        return f"${value / 1_000_000:,.1f}M"
    if abs(value) >= 1_000:
        return f"${value / 1_000:,.1f}K"
    return f"${value:,.2f}"


def format_number(value: int | float) -> str:
    """Format a number with commas."""
    if isinstance(value, float):
        return f"{value:,.2f}"
    return f"{value:,}"


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Convert a DataFrame to CSV bytes for download."""
    return df.to_csv(index=False).encode("utf-8")


def df_to_excel_bytes(df: pd.DataFrame) -> bytes:
    """Convert a DataFrame to Excel bytes for download."""
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Data")
    return buffer.getvalue()

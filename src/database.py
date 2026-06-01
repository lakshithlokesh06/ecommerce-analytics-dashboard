"""
database.py
-----------
SQLite operations for persistent data storage.
"""

import sqlite3
import os
import pandas as pd


DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ecommerce.db")


def get_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    """Get an SQLite connection."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def save_to_database(df: pd.DataFrame, table_name: str = "sales", db_path: str = DB_PATH):
    """Save a DataFrame to SQLite, replacing existing data."""
    conn = get_connection(db_path)
    try:
        df.to_sql(table_name, conn, if_exists="replace", index=False)
    finally:
        conn.close()


def load_from_database(table_name: str = "sales", db_path: str = DB_PATH) -> pd.DataFrame | None:
    """Load a table from SQLite.  Returns None if the table doesn't exist."""
    conn = get_connection(db_path)
    try:
        query = f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}';"
        exists = pd.read_sql(query, conn)
        if exists.empty:
            return None
        df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
        if "order_date" in df.columns:
            df["order_date"] = pd.to_datetime(df["order_date"])
        return df
    finally:
        conn.close()


def run_query(query: str, db_path: str = DB_PATH) -> pd.DataFrame:
    """Execute an arbitrary SQL query and return results."""
    conn = get_connection(db_path)
    try:
        return pd.read_sql(query, conn)
    finally:
        conn.close()

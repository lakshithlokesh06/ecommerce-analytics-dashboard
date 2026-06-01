"""
analytics.py
------------
Analytics computation functions for the dashboard.
"""

import pandas as pd
import numpy as np


def compute_kpis(df: pd.DataFrame) -> dict:
    total_revenue = df["revenue"].sum()
    total_orders = df["order_id"].nunique()
    avg_order_value = total_revenue / total_orders if total_orders else 0
    total_customers = df["customer_id"].nunique() if "customer_id" in df.columns else 0
    total_products = df["product_name"].nunique()
    avg_discount = df["discount"].mean() * 100 if "discount" in df.columns else 0
    return {
        "total_revenue": round(total_revenue, 2),
        "total_orders": total_orders,
        "avg_order_value": round(avg_order_value, 2),
        "total_customers": total_customers,
        "total_products": total_products,
        "avg_discount": round(avg_discount, 1),
    }


def monthly_revenue(df: pd.DataFrame) -> pd.DataFrame:
    m = df.groupby("year_month").agg(
        revenue=("revenue", "sum"), orders=("order_id", "nunique"), quantity=("quantity", "sum")
    ).reset_index().sort_values("year_month")
    m["revenue"] = m["revenue"].round(2)
    return m


def quarterly_revenue(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["year_quarter"] = df["year"].astype(str) + "-Q" + df["quarter"].astype(str)
    q = df.groupby("year_quarter").agg(
        revenue=("revenue", "sum"), orders=("order_id", "nunique")
    ).reset_index().sort_values("year_quarter")
    q["revenue"] = q["revenue"].round(2)
    return q


def daily_revenue(df: pd.DataFrame) -> pd.DataFrame:
    d = df.groupby(df["order_date"].dt.date).agg(
        revenue=("revenue", "sum"), orders=("order_id", "nunique")
    ).reset_index()
    d.columns = ["date", "revenue", "orders"]
    d["revenue"] = d["revenue"].round(2)
    return d.sort_values("date")


def top_products(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    t = df.groupby("product_name").agg(
        revenue=("revenue", "sum"), quantity=("quantity", "sum"), orders=("order_id", "nunique")
    ).reset_index().sort_values("revenue", ascending=False).head(n)
    t["revenue"] = t["revenue"].round(2)
    return t


def category_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    c = df.groupby("category").agg(
        revenue=("revenue", "sum"), quantity=("quantity", "sum"),
        orders=("order_id", "nunique"), avg_price=("unit_price", "mean")
    ).reset_index().sort_values("revenue", ascending=False)
    c["revenue"] = c["revenue"].round(2)
    c["avg_price"] = c["avg_price"].round(2)
    return c


def region_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    r = df.groupby("region").agg(
        revenue=("revenue", "sum"), orders=("order_id", "nunique"),
        customers=("customer_id", "nunique")
    ).reset_index().sort_values("revenue", ascending=False)
    r["revenue"] = r["revenue"].round(2)
    return r


def city_breakdown(df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    c = df.groupby(["region", "city"]).agg(
        revenue=("revenue", "sum"), orders=("order_id", "nunique")
    ).reset_index().sort_values("revenue", ascending=False).head(top_n)
    c["revenue"] = c["revenue"].round(2)
    return c


def payment_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    if "payment_method" not in df.columns:
        return pd.DataFrame()
    p = df.groupby("payment_method").agg(
        revenue=("revenue", "sum"), orders=("order_id", "nunique")
    ).reset_index().sort_values("revenue", ascending=False)
    p["revenue"] = p["revenue"].round(2)
    return p


def dow_analysis(df: pd.DataFrame) -> pd.DataFrame:
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    d = df.groupby("day_of_week").agg(
        revenue=("revenue", "sum"), orders=("order_id", "nunique")
    ).reindex(order).reset_index()
    d["revenue"] = d["revenue"].round(2)
    return d


def discount_impact(df: pd.DataFrame) -> pd.DataFrame:
    if "discount" not in df.columns:
        return pd.DataFrame()
    df = df.copy()
    bins = [0, 0.001, 0.10, 0.20, 0.30, 1.0]
    labels = ["No Discount", "1-10%", "11-20%", "21-30%", "30%+"]
    df["discount_tier"] = pd.cut(df["discount"], bins=bins, labels=labels, include_lowest=True)
    d = df.groupby("discount_tier", observed=False).agg(
        revenue=("revenue", "sum"), orders=("order_id", "nunique"),
        avg_quantity=("quantity", "mean")
    ).reset_index()
    d["revenue"] = d["revenue"].round(2)
    d["avg_quantity"] = d["avg_quantity"].round(2)
    return d

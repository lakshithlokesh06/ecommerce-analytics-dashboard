"""
E-Commerce Sales Analytics & Demand Forecasting Dashboard
==========================================================
Main Streamlit entry point — Home / Data Upload page.
"""

import streamlit as st
import pandas as pd
import os
import sys

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(__file__))

from src.data_loader import load_data, validate_columns, clean_data, get_cleaning_summary
from src.database import save_to_database, load_from_database

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="E-Commerce Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load CSS ─────────────────────────────────────────────────────────────────
css_path = os.path.join(os.path.dirname(__file__), "assets", "style.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛒 E-Commerce Analytics")
    st.markdown("---")
    st.markdown(
        """
        **Navigation**
        - 🏠 **Home** — Upload & Preview
        - 📊 **Analytics** — EDA & Insights
        - 🔮 **Forecasting** — ML Predictions

        ---
        Built with ❤️ using Streamlit
        """
    )

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("# 🛒 E-Commerce Sales Analytics & Demand Forecasting")
st.markdown(
    "*A comprehensive data analytics & data science portfolio project — "
    "upload your e-commerce dataset to explore insights and forecast demand.*"
)
st.markdown("---")

# ── Data Upload ──────────────────────────────────────────────────────────────
col_upload, col_info = st.columns([2, 1])

with col_upload:
    st.markdown("### 📂 Upload Your Dataset")
    uploaded_file = st.file_uploader(
        "Upload a CSV file with e-commerce sales data",
        type=["csv"],
        help="The CSV should contain columns like order_id, order_date, product_name, category, quantity, unit_price, revenue, region.",
    )

    # Also offer the sample dataset
    sample_path = os.path.join(os.path.dirname(__file__), "data", "ecommerce_sales.csv")
    use_sample = False
    if os.path.exists(sample_path):
        use_sample = st.button("📦 Use Sample Dataset (10K orders)", use_container_width=True)

with col_info:
    st.markdown("### 📋 Expected Columns")
    st.markdown(
        """
        | Column | Required |
        |--------|----------|
        | `order_id` | ✅ |
        | `order_date` | ✅ |
        | `product_name` | ✅ |
        | `category` | ✅ |
        | `quantity` | ✅ |
        | `unit_price` | ✅ |
        | `revenue` | ✅ |
        | `region` | ✅ |
        | `customer_id` | Optional |
        | `discount` | Optional |
        | `city` | Optional |
        | `payment_method` | Optional |
        """
    )

st.markdown("---")

# ── Process Data ─────────────────────────────────────────────────────────────
raw_df = None

if uploaded_file is not None:
    raw_df = load_data(uploaded_file)
elif use_sample:
    raw_df = load_data(sample_path)
else:
    # Check if data already exists in the database
    cached = load_from_database()
    if cached is not None:
        st.success("✅ Loaded previously saved data from database.")
        st.session_state["clean_df"] = cached
        raw_df = None  # skip cleaning

if raw_df is not None:
    # Validate
    missing_cols = validate_columns(raw_df)
    if missing_cols:
        st.error(f"❌ Missing required columns: **{', '.join(missing_cols)}**")
        st.stop()

    # Clean
    with st.spinner("🧹 Cleaning and processing data..."):
        clean_df = clean_data(raw_df)
        summary = get_cleaning_summary(raw_df, clean_df)
        save_to_database(clean_df)
        st.session_state["clean_df"] = clean_df

    # Show cleaning summary
    st.markdown("### 🧹 Data Cleaning Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows Before", f"{summary['rows_before']:,}")
    c2.metric("Rows After", f"{summary['rows_after']:,}")
    c3.metric("Duplicates Removed", f"{summary['duplicates_removed']:,}")
    c4.metric("Missing Values Fixed", f"{summary['missing_before']:,}")

# ── Data Preview ─────────────────────────────────────────────────────────────
if "clean_df" in st.session_state:
    df = st.session_state["clean_df"]

    st.markdown("### 👀 Data Preview")

    tab_head, tab_stats, tab_dtypes = st.tabs(["📄 Head", "📈 Statistics", "🏷️ Data Types"])

    with tab_head:
        st.dataframe(df.head(20), use_container_width=True, height=400)

    with tab_stats:
        st.dataframe(df.describe().round(2), use_container_width=True)

    with tab_dtypes:
        dtype_df = pd.DataFrame({
            "Column": df.columns,
            "Type": df.dtypes.astype(str).values,
            "Non-Null": df.notna().sum().values,
            "Null": df.isna().sum().values,
            "Unique": df.nunique().values,
        })
        st.dataframe(dtype_df, use_container_width=True)

    st.markdown("---")
    st.info("👉 Navigate to **📊 Analytics** or **🔮 Forecasting** from the sidebar to explore your data!")
else:
    st.markdown(
        """
        <div style="text-align:center; padding: 60px 20px;">
            <h2 style="color: #94a3b8;">📤 Upload a dataset to get started</h2>
            <p style="color: #64748b; font-size: 1.1em;">
                Or use the sample dataset with 10,000 e-commerce orders
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

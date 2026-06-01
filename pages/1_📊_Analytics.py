"""
📊 Analytics Dashboard
=======================
Interactive EDA with Plotly charts.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.analytics import (
    compute_kpis, monthly_revenue, quarterly_revenue, daily_revenue,
    top_products, category_breakdown, region_breakdown, city_breakdown,
    payment_breakdown, dow_analysis, discount_impact,
)
from src.database import load_from_database
from src.utils import format_currency, df_to_csv_bytes

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Analytics | E-Commerce", page_icon="📊", layout="wide")

css_path = os.path.join(os.path.dirname(__file__), "..", "assets", "style.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── Plotly theme ─────────────────────────────────────────────────────────────
COLORS = ["#6366f1", "#06b6d4", "#10b981", "#f59e0b", "#ef4444",
          "#ec4899", "#8b5cf6", "#14b8a6", "#f97316", "#64748b"]

PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", color="#e2e8f0"),
    xaxis=dict(gridcolor="#1e293b", zerolinecolor="#334155"),
    yaxis=dict(gridcolor="#1e293b", zerolinecolor="#334155"),
    margin=dict(l=40, r=20, t=50, b=40),
    hoverlabel=dict(bgcolor="#1e293b", font_color="#f1f5f9"),
)


def styled_chart(fig):
    """Apply consistent dark styling to a Plotly figure."""
    fig.update_layout(**PLOT_LAYOUT)
    return fig


# ── Load Data ────────────────────────────────────────────────────────────────
df = st.session_state.get("clean_df")
if df is None:
    df = load_from_database()

if df is None:
    st.warning("⚠️ No data found. Please upload a dataset on the **Home** page first.")
    st.stop()

# Ensure datetime
if not pd.api.types.is_datetime64_any_dtype(df["order_date"]):
    df["order_date"] = pd.to_datetime(df["order_date"])

# ── Sidebar Filters ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎛️ Filters")

    # Date range
    min_date = df["order_date"].min().date()
    max_date = df["order_date"].max().date()
    date_range = st.date_input("Date Range", value=(min_date, max_date),
                               min_value=min_date, max_value=max_date)

    # Category
    categories = ["All"] + sorted(df["category"].unique().tolist())
    sel_category = st.selectbox("Category", categories)

    # Region
    regions = ["All"] + sorted(df["region"].unique().tolist())
    sel_region = st.selectbox("Region", regions)

# Apply filters
mask = pd.Series(True, index=df.index)
if len(date_range) == 2:
    mask &= (df["order_date"].dt.date >= date_range[0]) & (df["order_date"].dt.date <= date_range[1])
if sel_category != "All":
    mask &= df["category"] == sel_category
if sel_region != "All":
    mask &= df["region"] == sel_region

filtered = df[mask].copy()

if filtered.empty:
    st.warning("No data matches the selected filters.")
    st.stop()

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("# 📊 Sales Analytics Dashboard")
st.markdown(f"*Showing **{len(filtered):,}** orders from **{filtered['order_date'].min().date()}** to **{filtered['order_date'].max().date()}***")
st.markdown("---")

# ── KPI Cards ────────────────────────────────────────────────────────────────
kpis = compute_kpis(filtered)
k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("💰 Total Revenue", format_currency(kpis["total_revenue"]))
k2.metric("📦 Total Orders", f"{kpis['total_orders']:,}")
k3.metric("🧾 Avg Order Value", format_currency(kpis["avg_order_value"]))
k4.metric("👥 Customers", f"{kpis['total_customers']:,}")
k5.metric("🏷️ Products", f"{kpis['total_products']:,}")
k6.metric("🏷️ Avg Discount", f"{kpis['avg_discount']}%")

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab_revenue, tab_products, tab_regions, tab_trends = st.tabs(
    ["💰 Revenue", "📦 Products", "🌍 Regions", "📈 Trends"]
)

# ── TAB 1: Revenue ───────────────────────────────────────────────────────────
with tab_revenue:
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### Monthly Revenue")
        mr = monthly_revenue(filtered)
        fig = px.bar(mr, x="year_month", y="revenue", text="revenue",
                     color_discrete_sequence=[COLORS[0]])
        fig.update_traces(texttemplate="$%{text:,.0f}", textposition="outside",
                          textfont_size=9)
        fig.update_layout(xaxis_title="Month", yaxis_title="Revenue ($)")
        st.plotly_chart(styled_chart(fig), use_container_width=True)

    with col_b:
        st.markdown("#### Quarterly Revenue")
        qr = quarterly_revenue(filtered)
        fig = px.bar(qr, x="year_quarter", y="revenue", text="revenue",
                     color_discrete_sequence=[COLORS[1]])
        fig.update_traces(texttemplate="$%{text:,.0f}", textposition="outside",
                          textfont_size=9)
        fig.update_layout(xaxis_title="Quarter", yaxis_title="Revenue ($)")
        st.plotly_chart(styled_chart(fig), use_container_width=True)

    # Daily trend line
    st.markdown("#### Daily Revenue Trend")
    dr = daily_revenue(filtered)
    fig = px.area(dr, x="date", y="revenue", color_discrete_sequence=[COLORS[0]])
    fig.update_traces(line=dict(width=1.5), fillcolor="rgba(99,102,241,0.15)")
    fig.update_layout(xaxis_title="Date", yaxis_title="Revenue ($)")
    st.plotly_chart(styled_chart(fig), use_container_width=True)

    # Payment & Discount
    col_c, col_d = st.columns(2)
    with col_c:
        st.markdown("#### Payment Methods")
        pay = payment_breakdown(filtered)
        if not pay.empty:
            fig = px.pie(pay, values="revenue", names="payment_method",
                         color_discrete_sequence=COLORS, hole=0.45)
            fig.update_traces(textinfo="label+percent", textfont_size=11)
            st.plotly_chart(styled_chart(fig), use_container_width=True)

    with col_d:
        st.markdown("#### Discount Impact on Revenue")
        disc = discount_impact(filtered)
        if not disc.empty:
            fig = px.bar(disc, x="discount_tier", y="revenue", text="revenue",
                         color="discount_tier", color_discrete_sequence=COLORS)
            fig.update_traces(texttemplate="$%{text:,.0f}", textposition="outside",
                              textfont_size=9)
            fig.update_layout(showlegend=False, xaxis_title="Discount Tier",
                              yaxis_title="Revenue ($)")
            st.plotly_chart(styled_chart(fig), use_container_width=True)

# ── TAB 2: Products ──────────────────────────────────────────────────────────
with tab_products:
    col_e, col_f = st.columns(2)

    with col_e:
        st.markdown("#### Top 10 Products by Revenue")
        tp = top_products(filtered, 10)
        fig = px.bar(tp, y="product_name", x="revenue", orientation="h",
                     text="revenue", color="revenue",
                     color_continuous_scale="Viridis")
        fig.update_traces(texttemplate="$%{text:,.0f}", textposition="outside",
                          textfont_size=9)
        fig.update_layout(yaxis=dict(autorange="reversed"), coloraxis_showscale=False,
                          xaxis_title="Revenue ($)", yaxis_title="")
        st.plotly_chart(styled_chart(fig), use_container_width=True)

    with col_f:
        st.markdown("#### Category Revenue Share")
        cb = category_breakdown(filtered)
        fig = px.pie(cb, values="revenue", names="category",
                     color_discrete_sequence=COLORS, hole=0.45)
        fig.update_traces(textinfo="label+percent+value", textfont_size=11,
                          texttemplate="%{label}<br>%{percent}<br>$%{value:,.0f}")
        st.plotly_chart(styled_chart(fig), use_container_width=True)

    # Category detail table
    st.markdown("#### Category Breakdown")
    st.dataframe(cb, use_container_width=True)

# ── TAB 3: Regions ───────────────────────────────────────────────────────────
with tab_regions:
    col_g, col_h = st.columns(2)

    with col_g:
        st.markdown("#### Revenue by Region")
        rb = region_breakdown(filtered)
        fig = px.bar(rb, x="region", y="revenue", text="revenue",
                     color="region", color_discrete_sequence=COLORS)
        fig.update_traces(texttemplate="$%{text:,.0f}", textposition="outside",
                          textfont_size=10)
        fig.update_layout(showlegend=False, xaxis_title="Region", yaxis_title="Revenue ($)")
        st.plotly_chart(styled_chart(fig), use_container_width=True)

    with col_h:
        st.markdown("#### Top 15 Cities")
        ct = city_breakdown(filtered, 15)
        fig = px.bar(ct, y="city", x="revenue", orientation="h", text="revenue",
                     color="region", color_discrete_sequence=COLORS)
        fig.update_traces(texttemplate="$%{text:,.0f}", textposition="outside",
                          textfont_size=9)
        fig.update_layout(yaxis=dict(autorange="reversed"),
                          xaxis_title="Revenue ($)", yaxis_title="")
        st.plotly_chart(styled_chart(fig), use_container_width=True)

    # Treemap
    st.markdown("#### Region → City Treemap")
    tree_data = filtered.groupby(["region", "city"]).agg(
        revenue=("revenue", "sum")).reset_index()
    fig = px.treemap(tree_data, path=["region", "city"], values="revenue",
                     color="revenue", color_continuous_scale="Viridis")
    fig.update_layout(margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(styled_chart(fig), use_container_width=True)

# ── TAB 4: Trends ────────────────────────────────────────────────────────────
with tab_trends:
    col_i, col_j = st.columns(2)

    with col_i:
        st.markdown("#### Orders by Day of Week")
        dw = dow_analysis(filtered)
        fig = px.bar(dw, x="day_of_week", y="orders", text="orders",
                     color_discrete_sequence=[COLORS[2]])
        fig.update_traces(textposition="outside", textfont_size=10)
        fig.update_layout(xaxis_title="Day", yaxis_title="Orders")
        st.plotly_chart(styled_chart(fig), use_container_width=True)

    with col_j:
        st.markdown("#### Revenue by Day of Week")
        fig = px.bar(dw, x="day_of_week", y="revenue", text="revenue",
                     color_discrete_sequence=[COLORS[3]])
        fig.update_traces(texttemplate="$%{text:,.0f}", textposition="outside",
                          textfont_size=10)
        fig.update_layout(xaxis_title="Day", yaxis_title="Revenue ($)")
        st.plotly_chart(styled_chart(fig), use_container_width=True)

    # Monthly orders + revenue overlay
    st.markdown("#### Monthly Orders vs Revenue")
    mr = monthly_revenue(filtered)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=mr["year_month"], y=mr["orders"], name="Orders",
                         marker_color=COLORS[0], yaxis="y"))
    fig.add_trace(go.Scatter(x=mr["year_month"], y=mr["revenue"], name="Revenue",
                             line=dict(color=COLORS[3], width=3), yaxis="y2"))
    fig.update_layout(
        yaxis=dict(title="Orders", side="left"),
        yaxis2=dict(title="Revenue ($)", side="right", overlaying="y"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )
    st.plotly_chart(styled_chart(fig), use_container_width=True)

# ── Download ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 📥 Download Filtered Data")
st.download_button(
    "⬇️ Download as CSV",
    data=df_to_csv_bytes(filtered),
    file_name="ecommerce_analytics_filtered.csv",
    mime="text/csv",
    use_container_width=True,
)

"""
🔮 Forecasting Dashboard
==========================
ML model training, evaluation, and demand forecasting.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.forecasting import (
    prepare_features, train_models, get_predictions_df,
    get_feature_importance, forecast_monthly_revenue,
)
from src.database import load_from_database
from src.utils import format_currency, df_to_csv_bytes

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Forecasting | E-Commerce", page_icon="🔮", layout="wide")

css_path = os.path.join(os.path.dirname(__file__), "..", "assets", "style.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

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
    fig.update_layout(**PLOT_LAYOUT)
    return fig


# ── Load Data ────────────────────────────────────────────────────────────────
df = st.session_state.get("clean_df")
if df is None:
    df = load_from_database()

if df is None:
    st.warning("⚠️ No data found. Please upload a dataset on the **Home** page first.")
    st.stop()

if not pd.api.types.is_datetime64_any_dtype(df["order_date"]):
    df["order_date"] = pd.to_datetime(df["order_date"])

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("# 🔮 Demand Forecasting & Sales Prediction")
st.markdown("*Train ML models on your sales data and forecast future revenue.*")
st.markdown("---")

# ── Sidebar Controls ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Model Settings")
    test_size = st.slider("Test Set Size", 0.1, 0.4, 0.2, 0.05)
    forecast_months = st.slider("Forecast Months Ahead", 1, 12, 6)
    st.markdown("---")

# ── Train Models ─────────────────────────────────────────────────────────────
st.markdown("### 🤖 Model Training")
st.markdown(f"Using **{int((1-test_size)*100)}%** training / **{int(test_size*100)}%** test split")

if st.button("🚀 Train Models", use_container_width=True, type="primary"):
    with st.spinner("⏳ Engineering features & training models... This may take a moment."):
        X, y = prepare_features(df)
        trained_models, results, X_test, y_test = train_models(X, y, test_size)

        st.session_state["trained_models"] = trained_models
        st.session_state["model_results"] = results
        st.session_state["X_test"] = X_test
        st.session_state["y_test"] = y_test
        st.session_state["feature_names"] = list(X.columns)

    st.success("✅ All models trained successfully!")

# ── Show Results ─────────────────────────────────────────────────────────────
if "model_results" in st.session_state:
    results = st.session_state["model_results"]
    trained_models = st.session_state["trained_models"]
    X_test = st.session_state["X_test"]
    y_test = st.session_state["y_test"]
    feature_names = st.session_state["feature_names"]

    # ── Metrics Comparison ───────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📏 Model Evaluation Metrics")

    metrics_df = pd.DataFrame(results).T.reset_index()
    metrics_df.columns = ["Model", "MAE", "RMSE", "R²"]

    # KPI cards for best model
    best_model_name = metrics_df.loc[metrics_df["R²"].idxmax(), "Model"]
    best = results[best_model_name]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("🏆 Best Model", best_model_name)
    m2.metric("MAE", f"${best['MAE']:,.2f}")
    m3.metric("RMSE", f"${best['RMSE']:,.2f}")
    m4.metric("R² Score", f"{best['R²']:.4f}")

    # Metrics bar chart
    col_met1, col_met2 = st.columns(2)
    with col_met1:
        fig = px.bar(metrics_df, x="Model", y=["MAE", "RMSE"], barmode="group",
                     color_discrete_sequence=[COLORS[0], COLORS[4]],
                     title="MAE & RMSE by Model (lower is better)")
        st.plotly_chart(styled_chart(fig), use_container_width=True)

    with col_met2:
        fig = px.bar(metrics_df, x="Model", y="R²", text="R²",
                     color_discrete_sequence=[COLORS[2]],
                     title="R² Score by Model (higher is better)")
        fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
        st.plotly_chart(styled_chart(fig), use_container_width=True)

    # Full metrics table
    st.dataframe(metrics_df.style.highlight_max(subset=["R²"], color="#10b981")
                 .highlight_min(subset=["MAE", "RMSE"], color="#10b981"),
                 use_container_width=True)

    # ── Actual vs Predicted ──────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🎯 Actual vs Predicted")

    sel_model = st.selectbox("Select Model", list(trained_models.keys()),
                             index=list(trained_models.keys()).index(best_model_name))
    model = trained_models[sel_model]
    pred_df = get_predictions_df(model, X_test, y_test)

    col_p1, col_p2 = st.columns(2)
    with col_p1:
        # Scatter plot
        fig = px.scatter(pred_df, x="Actual", y="Predicted",
                         color_discrete_sequence=[COLORS[0]],
                         title=f"{sel_model}: Actual vs Predicted",
                         opacity=0.5)
        # Perfect prediction line
        min_val = min(pred_df["Actual"].min(), pred_df["Predicted"].min())
        max_val = max(pred_df["Actual"].max(), pred_df["Predicted"].max())
        fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                                 mode="lines", name="Perfect Prediction",
                                 line=dict(color="#ef4444", dash="dash", width=2)))
        fig.update_layout(xaxis_title="Actual Revenue ($)",
                          yaxis_title="Predicted Revenue ($)")
        st.plotly_chart(styled_chart(fig), use_container_width=True)

    with col_p2:
        # Residual distribution
        fig = px.histogram(pred_df, x="Residual", nbins=50,
                           color_discrete_sequence=[COLORS[1]],
                           title="Residual Distribution")
        fig.update_layout(xaxis_title="Residual ($)", yaxis_title="Count")
        st.plotly_chart(styled_chart(fig), use_container_width=True)

    # ── Feature Importance ───────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🧬 Feature Importance")
    imp_df = get_feature_importance(model, feature_names)
    if not imp_df.empty:
        fig = px.bar(imp_df, y="Feature", x="Importance", orientation="h",
                     color="Importance", color_continuous_scale="Viridis",
                     title=f"{sel_model} — Feature Importance")
        fig.update_layout(yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
        st.plotly_chart(styled_chart(fig), use_container_width=True)
    else:
        st.info("Feature importance is only available for tree-based models.")

    # ── Future Forecast ──────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(f"### 🔮 {forecast_months}-Month Revenue Forecast")

    with st.spinner("Generating forecast..."):
        forecast = forecast_monthly_revenue(df, model, periods=forecast_months)

    # Historical + Forecast chart
    hist = df.groupby("year_month").agg(revenue=("revenue", "sum")).reset_index()
    hist.columns = ["Month", "Revenue"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist["Month"], y=hist["Revenue"],
                             mode="lines+markers", name="Historical",
                             line=dict(color=COLORS[0], width=2.5),
                             marker=dict(size=6)))
    fig.add_trace(go.Scatter(x=forecast["Month"], y=forecast["Forecasted Revenue"],
                             mode="lines+markers", name="Forecast",
                             line=dict(color=COLORS[3], width=2.5, dash="dot"),
                             marker=dict(size=8, symbol="diamond")))
    fig.update_layout(title="Historical Revenue & Future Forecast",
                      xaxis_title="Month", yaxis_title="Revenue ($)",
                      legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center"))
    st.plotly_chart(styled_chart(fig), use_container_width=True)

    # Forecast table
    st.dataframe(forecast, use_container_width=True)

    # ── Downloads ────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📥 Download Results")
    dl1, dl2, dl3 = st.columns(3)
    with dl1:
        st.download_button("⬇️ Predictions CSV", df_to_csv_bytes(pred_df),
                           "predictions.csv", "text/csv", use_container_width=True)
    with dl2:
        st.download_button("⬇️ Forecast CSV", df_to_csv_bytes(forecast),
                           "forecast.csv", "text/csv", use_container_width=True)
    with dl3:
        st.download_button("⬇️ Metrics CSV", df_to_csv_bytes(metrics_df),
                           "model_metrics.csv", "text/csv", use_container_width=True)

else:
    st.markdown(
        """
        <div style="text-align:center; padding: 60px 20px;">
            <h2 style="color: #94a3b8;">🚀 Click "Train Models" to get started</h2>
            <p style="color: #64748b; font-size: 1.1em;">
                The system will train Random Forest, Gradient Boosting, and Linear Regression
                models on your dataset and display evaluation metrics and forecasts.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

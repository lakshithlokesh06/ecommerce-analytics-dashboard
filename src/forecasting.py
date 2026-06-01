"""
forecasting.py
--------------
ML-based demand forecasting pipeline.
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Engineer features for the ML models."""
    feat = df.copy()

    # Numeric time features
    feat["month"] = feat["order_date"].dt.month
    feat["day"] = feat["order_date"].dt.day
    feat["day_of_week_num"] = feat["order_date"].dt.dayofweek
    feat["week_of_year"] = feat["order_date"].dt.isocalendar().week.astype(int)
    feat["is_weekend"] = (feat["day_of_week_num"] >= 5).astype(int)

    # Cyclical encoding for month
    feat["month_sin"] = np.sin(2 * np.pi * feat["month"] / 12)
    feat["month_cos"] = np.cos(2 * np.pi * feat["month"] / 12)

    # Encode categoricals
    le_cols = {}
    for col in ["category", "region"]:
        if col in feat.columns:
            le = LabelEncoder()
            feat[f"{col}_encoded"] = le.fit_transform(feat[col].astype(str))
            le_cols[col] = le

    feature_cols = [
        "quantity", "unit_price", "month", "day", "day_of_week_num",
        "week_of_year", "is_weekend", "month_sin", "month_cos",
    ]
    if "discount" in feat.columns:
        feature_cols.append("discount")
    if "category_encoded" in feat.columns:
        feature_cols.append("category_encoded")
    if "region_encoded" in feat.columns:
        feature_cols.append("region_encoded")

    X = feat[feature_cols].fillna(0)
    y = feat["revenue"]
    return X, y


def train_models(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2):
    """Train multiple models and return results."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    models = {
        "Random Forest": RandomForestRegressor(
            n_estimators=100, max_depth=12, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42
        ),
        "Linear Regression": LinearRegression(),
    }

    results = {}
    trained = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        results[name] = {
            "MAE": round(mae, 2),
            "RMSE": round(rmse, 2),
            "R²": round(r2, 4),
        }
        trained[name] = model

        # Save model
        path = os.path.join(MODELS_DIR, f"{name.lower().replace(' ', '_')}.pkl")
        with open(path, "wb") as f:
            pickle.dump(model, f)

    return trained, results, X_test, y_test


def get_predictions_df(model, X_test, y_test) -> pd.DataFrame:
    """Return a DataFrame of actual vs predicted values."""
    y_pred = model.predict(X_test)
    pred_df = pd.DataFrame({
        "Actual": y_test.values,
        "Predicted": np.round(y_pred, 2),
        "Residual": np.round(y_test.values - y_pred, 2),
    })
    return pred_df


def get_feature_importance(model, feature_names: list) -> pd.DataFrame:
    """Extract feature importances (tree-based models only)."""
    if hasattr(model, "feature_importances_"):
        imp = pd.DataFrame({
            "Feature": feature_names,
            "Importance": model.feature_importances_,
        }).sort_values("Importance", ascending=False)
        return imp
    return pd.DataFrame()


def forecast_monthly_revenue(df: pd.DataFrame, model, periods: int = 6):
    """Forecast future monthly revenue."""
    monthly = df.groupby("year_month").agg(
        revenue=("revenue", "sum"),
        avg_quantity=("quantity", "mean"),
        avg_price=("unit_price", "mean"),
        avg_discount=("discount", "mean") if "discount" in df.columns else ("revenue", "count"),
    ).reset_index()

    last_date = df["order_date"].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=periods, freq="MS")

    future_records = []
    for d in future_dates:
        future_records.append({
            "quantity": monthly["avg_quantity"].mean(),
            "unit_price": monthly["avg_price"].mean(),
            "month": d.month,
            "day": 15,
            "day_of_week_num": d.dayofweek,
            "week_of_year": d.isocalendar()[1],
            "is_weekend": 0,
            "month_sin": np.sin(2 * np.pi * d.month / 12),
            "month_cos": np.cos(2 * np.pi * d.month / 12),
            "discount": monthly["avg_discount"].mean() if "avg_discount" in monthly.columns else 0,
            "category_encoded": 0,
            "region_encoded": 0,
        })

    future_df = pd.DataFrame(future_records)
    # Match columns to model's training features
    X_cols = model.feature_names_in_ if hasattr(model, "feature_names_in_") else future_df.columns
    for col in X_cols:
        if col not in future_df.columns:
            future_df[col] = 0
    future_df = future_df[X_cols]

    predictions = model.predict(future_df)

    # Scale single-order predictions to monthly level
    avg_orders_per_month = len(df) / df["year_month"].nunique()
    monthly_predictions = predictions * avg_orders_per_month

    forecast_df = pd.DataFrame({
        "Month": future_dates.strftime("%Y-%m"),
        "Forecasted Revenue": np.round(monthly_predictions, 2),
    })
    return forecast_df

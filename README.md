# 🛒 E-Commerce Sales Analytics & Demand Forecasting Dashboard

A comprehensive **Data Analytics** and **Data Science** portfolio project built with **Streamlit**, featuring interactive sales analytics, EDA, and ML-powered demand forecasting.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.45-red?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🎯 Project Overview

This project demonstrates end-to-end data analytics and data science skills:

| Skill Area | What's Covered |
|-----------|---------------|
| **Data Cleaning** | Missing values, duplicates, type enforcement |
| **EDA** | Revenue trends, product analysis, region breakdowns |
| **Visualization** | 15+ interactive Plotly charts with dark theme |
| **Machine Learning** | Random Forest, Gradient Boosting, Linear Regression |
| **Model Evaluation** | MAE, RMSE, R² metrics with comparisons |
| **Forecasting** | Monthly revenue predictions for future periods |
| **Database** | SQLite for persistent data storage |
| **Web App** | Multi-page Streamlit dashboard with custom CSS |

---

## 🏗️ Architecture

```
CSV Upload → Data Cleaning → SQLite Storage
                                    ↓
                    ┌───────────────┼───────────────┐
                    ↓               ↓               ↓
              Revenue EDA    Product Analysis   Region Analysis
                    ↓               ↓               ↓
                    └───────────────┼───────────────┘
                                    ↓
                          Feature Engineering
                                    ↓
                    ┌───────────────┼───────────────┐
                    ↓               ↓               ↓
              Random Forest   Gradient Boost   Linear Regression
                    ↓               ↓               ↓
                    └───────────────┼───────────────┘
                                    ↓
                    Model Evaluation (MAE, RMSE, R²)
                                    ↓
                        Demand Forecasting
                                    ↓
                      Interactive Dashboard
```

---

## 📁 Folder Structure

```
ecommerce-analytics-dashboard/
├── app.py                          # Main entry point
├── pages/
│   ├── 1_📊_Analytics.py           # Analytics dashboard
│   └── 2_🔮_Forecasting.py        # Forecasting dashboard
├── src/
│   ├── __init__.py
│   ├── data_loader.py              # Data cleaning pipeline
│   ├── analytics.py                # Analytics computations
│   ├── forecasting.py              # ML models
│   ├── database.py                 # SQLite layer
│   └── utils.py                    # Helpers
├── data/
│   └── generate_sample_data.py     # Sample data generator
├── assets/
│   └── style.css                   # Custom theme
├── models/                         # Saved ML models
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/ecommerce-analytics-dashboard.git
cd ecommerce-analytics-dashboard

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Generate Sample Data

```bash
python data/generate_sample_data.py
```

This creates `data/ecommerce_sales.csv` with **10,000** realistic e-commerce orders.

### Run the Dashboard

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## 📊 Dashboard Pages

### 🏠 Home — Data Upload
- Upload CSV or use sample dataset
- Automatic data cleaning & validation
- Data preview with statistics

### 📊 Analytics
- **KPI Cards**: Revenue, Orders, AOV, Customers, Products, Avg Discount
- **Revenue Tab**: Monthly, Quarterly, Daily trends, Payment methods, Discount impact
- **Products Tab**: Top 10 products, Category revenue share
- **Regions Tab**: Region breakdown, Top cities, Treemap
- **Trends Tab**: Day-of-week patterns, Orders vs Revenue overlay
- Sidebar filters: Date range, Category, Region

### 🔮 Forecasting
- Train 3 ML models with configurable test split
- Evaluation metrics: MAE, RMSE, R²
- Actual vs Predicted scatter plot
- Residual distribution
- Feature importance chart
- N-month revenue forecast
- Download predictions, forecasts, and metrics as CSV

---

## 🔧 Technology Stack

| Technology | Purpose |
|-----------|---------|
| Python 3.10+ | Core language |
| Streamlit | Web dashboard |
| Pandas & NumPy | Data manipulation |
| Plotly | Interactive charts |
| Scikit-learn | ML models |
| SQLite | Data persistence |
| openpyxl | Excel export |

---

## 📊 Dataset Schema

| Column | Type | Description |
|--------|------|-------------|
| order_id | str | Unique order ID |
| order_date | date | Order date |
| customer_id | str | Customer ID |
| product_name | str | Product name |
| category | str | Product category |
| quantity | int | Units ordered |
| unit_price | float | Price per unit |
| discount | float | Discount (0-0.35) |
| revenue | float | Net revenue |
| region | str | Geographic region |
| city | str | City |
| payment_method | str | Payment type |
| shipping_cost | float | Shipping fee |

---

## 🤖 ML Models

| Model | Description |
|-------|-------------|
| **Random Forest** | Ensemble of 100 decision trees, max depth 12 |
| **Gradient Boosting** | Sequential boosting with 100 estimators, LR 0.1 |
| **Linear Regression** | Baseline linear model |

**Features engineered**: month, day, day_of_week, week_of_year, is_weekend, cyclical month encoding, category/region encoding, quantity, unit_price, discount.

---

## 📦 GitHub Setup

```bash
cd ecommerce-analytics-dashboard
git init
git add .
git commit -m "Initial commit: E-Commerce Analytics Dashboard"
git branch -M main
git remote add origin https://github.com/<username>/ecommerce-analytics-dashboard.git
git push -u origin main
```

---

## 📄 License

This project is open source under the [MIT License](LICENSE).

---

## 🙏 Acknowledgments

- [Streamlit](https://streamlit.io/) for the amazing web framework
- [Plotly](https://plotly.com/) for interactive visualizations
- [Scikit-learn](https://scikit-learn.org/) for ML algorithms

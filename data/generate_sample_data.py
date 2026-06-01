"""
generate_sample_data.py
-----------------------
Generates a realistic synthetic e-commerce dataset (10,000 orders spanning
2 years) and saves it as data/ecommerce_sales.csv.
"""

import os
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

random.seed(42)
np.random.seed(42)

# ── Product Catalog ──────────────────────────────────────────────────────────
PRODUCTS = {
    "Electronics": [
        ("Wireless Bluetooth Headphones", 79.99),
        ("Smartphone Case Premium", 29.99),
        ("USB-C Fast Charger", 24.99),
        ("Portable Power Bank 20000mAh", 49.99),
        ("Mechanical Gaming Keyboard", 129.99),
        ("Wireless Gaming Mouse", 59.99),
        ("4K HDMI Cable 6ft", 14.99),
        ("Smart Watch Fitness Tracker", 199.99),
        ("Noise Cancelling Earbuds", 149.99),
        ("Laptop Stand Aluminum", 45.99),
    ],
    "Clothing": [
        ("Cotton Crew Neck T-Shirt", 19.99),
        ("Slim Fit Denim Jeans", 49.99),
        ("Running Shoes Pro", 89.99),
        ("Winter Puffer Jacket", 129.99),
        ("Casual Sneakers", 64.99),
        ("Formal Dress Shirt", 39.99),
        ("Yoga Pants Flex", 34.99),
        ("Wool Blend Sweater", 59.99),
        ("Athletic Shorts", 24.99),
        ("Leather Belt Classic", 29.99),
    ],
    "Home & Kitchen": [
        ("Stainless Steel Water Bottle", 22.99),
        ("Non-Stick Cookware Set", 89.99),
        ("Memory Foam Pillow", 39.99),
        ("LED Desk Lamp", 34.99),
        ("Coffee Maker Programmable", 69.99),
        ("Air Purifier HEPA", 149.99),
        ("Robot Vacuum Cleaner", 249.99),
        ("Bamboo Cutting Board Set", 27.99),
        ("Scented Candle Gift Set", 24.99),
        ("Smart Thermostat", 179.99),
    ],
    "Books & Media": [
        ("Python Data Science Handbook", 44.99),
        ("Machine Learning Yearning", 29.99),
        ("The Lean Startup", 16.99),
        ("Deep Work", 14.99),
        ("Atomic Habits", 18.99),
        ("Thinking Fast and Slow", 17.99),
        ("Designing Data-Intensive Apps", 49.99),
        ("Clean Code", 39.99),
        ("The Pragmatic Programmer", 49.99),
        ("Storytelling with Data", 32.99),
    ],
    "Sports & Outdoors": [
        ("Yoga Mat Premium", 34.99),
        ("Resistance Bands Set", 19.99),
        ("Camping Tent 4-Person", 159.99),
        ("Hiking Backpack 50L", 79.99),
        ("Dumbbell Set Adjustable", 129.99),
        ("Cycling Helmet", 49.99),
        ("Jump Rope Speed", 12.99),
        ("Foam Roller Recovery", 24.99),
        ("Insulated Water Jug 1gal", 29.99),
        ("Fishing Rod Combo", 69.99),
    ],
}

REGIONS = {
    "North": ["New York", "Boston", "Chicago", "Detroit", "Minneapolis"],
    "South": ["Houston", "Dallas", "Miami", "Atlanta", "Charlotte"],
    "East": ["Philadelphia", "Baltimore", "Richmond", "Pittsburgh", "Newark"],
    "West": ["Los Angeles", "San Francisco", "Seattle", "Denver", "Phoenix"],
    "Central": ["Kansas City", "St. Louis", "Nashville", "Indianapolis", "Columbus"],
}

PAYMENT_METHODS = ["Credit Card", "Debit Card", "PayPal", "UPI", "Net Banking"]

FIRST_NAMES = [
    "James", "Mary", "Robert", "Patricia", "John", "Jennifer", "Michael",
    "Linda", "David", "Elizabeth", "William", "Barbara", "Richard", "Susan",
    "Joseph", "Jessica", "Thomas", "Sarah", "Charles", "Karen", "Arun",
    "Priya", "Raj", "Anita", "Vikram", "Deepa", "Suresh", "Meena",
    "Amit", "Kavitha", "Wei", "Yuki", "Chen", "Sakura", "Hiroshi",
]

LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
    "Davis", "Rodriguez", "Martinez", "Anderson", "Taylor", "Thomas",
    "Hernandez", "Moore", "Patel", "Kumar", "Sharma", "Singh", "Gupta",
    "Reddy", "Nair", "Wang", "Li", "Zhang", "Tanaka", "Sato", "Kim",
]


def generate_dataset(n_orders: int = 10000) -> pd.DataFrame:
    """Generate *n_orders* synthetic e-commerce transactions."""

    start_date = datetime(2023, 1, 1)
    end_date = datetime(2024, 12, 31)
    date_range_days = (end_date - start_date).days

    records = []
    customer_pool = [
        f"CUST-{i:05d}" for i in range(1, 2001)
    ]
    customer_names = {
        cid: f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"
        for cid in customer_pool
    }

    for i in range(1, n_orders + 1):
        # Seasonal weight: higher sales in Nov-Dec (holiday) and Jun-Jul (summer)
        order_date = start_date + timedelta(days=random.randint(0, date_range_days))
        month = order_date.month
        seasonal_boost = 1.0
        if month in (11, 12):
            seasonal_boost = 1.5
        elif month in (6, 7):
            seasonal_boost = 1.2

        category = random.choices(
            list(PRODUCTS.keys()),
            weights=[0.30, 0.25, 0.20, 0.10, 0.15],
            k=1,
        )[0]
        product_name, unit_price = random.choice(PRODUCTS[category])
        product_id = f"SKU-{abs(hash(product_name)) % 100000:05d}"

        quantity = max(1, int(np.random.exponential(2) * seasonal_boost))
        quantity = min(quantity, 15)  # cap

        discount = random.choices(
            [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
            weights=[0.35, 0.20, 0.15, 0.12, 0.08, 0.05, 0.05],
            k=1,
        )[0]

        revenue = round(quantity * unit_price * (1 - discount), 2)

        region = random.choice(list(REGIONS.keys()))
        city = random.choice(REGIONS[region])

        customer_id = random.choice(customer_pool)

        shipping_cost = round(random.uniform(3.99, 15.99), 2)

        records.append(
            {
                "order_id": f"ORD-{i:06d}",
                "order_date": order_date.strftime("%Y-%m-%d"),
                "customer_id": customer_id,
                "customer_name": customer_names[customer_id],
                "product_id": product_id,
                "product_name": product_name,
                "category": category,
                "quantity": quantity,
                "unit_price": unit_price,
                "discount": discount,
                "revenue": revenue,
                "region": region,
                "city": city,
                "payment_method": random.choice(PAYMENT_METHODS),
                "shipping_cost": shipping_cost,
            }
        )

    df = pd.DataFrame(records)

    # Introduce ~2 % missing values for realistic cleaning demo
    n_missing = int(len(df) * 0.02)
    for col in ["customer_name", "city", "discount", "shipping_cost"]:
        idx = df.sample(n=n_missing, random_state=42).index
        df.loc[idx, col] = np.nan

    return df


if __name__ == "__main__":
    output_dir = os.path.join(os.path.dirname(__file__))
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "ecommerce_sales.csv")

    df = generate_dataset(10000)
    df.to_csv(output_path, index=False)
    print(f"✅  Generated {len(df):,} orders → {output_path}")
    print(f"    Columns : {list(df.columns)}")
    print(f"    Date range : {df['order_date'].min()} → {df['order_date'].max()}")
    print(f"    Missing values :\n{df.isnull().sum()[df.isnull().sum() > 0]}")

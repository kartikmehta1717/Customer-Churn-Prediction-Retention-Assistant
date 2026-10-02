import pandas as pd
from sqlalchemy import text
from src.db import get_engine

df = pd.read_csv("data/processed/telco_cleaned.csv")
print("Columns:", df.columns.tolist())
print("Shape:", df.shape)

# MySQL / Power BI friendly column names
df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

# Add original Customer ID from raw dataset
raw_df = pd.read_csv("data/raw/telco.csv")
df.insert(0, "customer_id", raw_df["Customer ID"])

engine = get_engine()
df.to_sql("customers", engine, if_exists="replace", index=False)

with engine.connect() as conn:
    n = conn.execute(text("SELECT COUNT(*) FROM customers")).scalar()
    print("Rows in MySQL:", n)
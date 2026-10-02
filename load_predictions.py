import pandas as pd
from sqlalchemy import text
from src.db import get_engine

# Load predictions
df = pd.read_csv("data/processed/predictions.csv")

print("Columns:", df.columns.tolist())
print("Shape:", df.shape)

# Connect to MySQL
engine = get_engine()

# Load into MySQL
df.to_sql(
    "predictions",
    engine,
    if_exists="replace",
    index=False
)

# Verify
with engine.connect() as conn:
    count = conn.execute(
        text("SELECT COUNT(*) FROM predictions")
    ).scalar()

    print("Rows in MySQL predictions table:", count)
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split


# -----------------------------
# Paths
# -----------------------------
RAW_PATH = "data/raw/telco.csv"
X_TEST_PATH = "data/processed/X_test.csv"
Y_TEST_PATH = "data/processed/y_test.csv"
MODEL_PATH = "models/churn_model.pkl"
THRESHOLD_PATH = "models/best_threshold.pkl"


# -----------------------------
# Load model + test data
# -----------------------------
model = joblib.load(MODEL_PATH)
best_threshold = float(joblib.load(THRESHOLD_PATH))

X_test = pd.read_csv(X_TEST_PATH)
y_test = pd.read_csv(Y_TEST_PATH).squeeze()

print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)
print("Threshold:", best_threshold)


# -----------------------------
# Recover original Customer IDs
# using the SAME split as training
# -----------------------------
source = pd.read_csv(RAW_PATH)

source_target = source["Churn Label"].map({
    "Yes": 1,
    "No": 0
})

_, test_rows = train_test_split(
    np.arange(len(source)),
    test_size=0.2,
    random_state=42,
    stratify=source_target
)

customer_ids = (
    source.iloc[test_rows]["Customer ID"]
    .reset_index(drop=True)
)


# -----------------------------
# Safety checks
# -----------------------------
if len(customer_ids) != len(X_test):
    raise ValueError(
        f"ID count ({len(customer_ids)}) != X_test rows ({len(X_test)})"
    )

if len(y_test) != len(X_test):
    raise ValueError(
        f"y_test rows ({len(y_test)}) != X_test rows ({len(X_test)})"
    )

target_check = source_target.iloc[test_rows].reset_index(drop=True)

if not np.array_equal(target_check.values, y_test.values):
    raise ValueError(
        "Customer ID mapping does not match y_test!"
    )


# -----------------------------
# Generate predictions
# -----------------------------
probabilities = model.predict_proba(X_test)[:, 1]

predictions = (
    probabilities >= best_threshold
).astype(int)


# -----------------------------
# Create prediction table
# -----------------------------
predictions_df = pd.DataFrame({
    "customer_id": customer_ids,
    "churn_probability": probabilities,
    "prediction": predictions,
    "prediction_label": np.where(
        predictions == 1,
        "Yes",
        "No"
    )
})


# -----------------------------
# Risk level
# -----------------------------
predictions_df["risk_level"] = np.select(
    [
        predictions_df["churn_probability"] >= 0.6,
        predictions_df["churn_probability"] >= 0.4
    ],
    [
        "High",
        "Medium"
    ],
    default="Low"
)


# -----------------------------
# Save locally
# -----------------------------
output_path = "data/processed/predictions.csv"

predictions_df.to_csv(
    output_path,
    index=False
)


# -----------------------------
# Verification
# -----------------------------
print("\nPrediction table created successfully!")
print("Rows:", len(predictions_df))
print("\nFirst 5 rows:")
print(predictions_df.head())

print("\nPrediction distribution:")
print(predictions_df["prediction_label"].value_counts())

print("\nRisk distribution:")
print(predictions_df["risk_level"].value_counts())

print(f"\nSaved to: {output_path}")
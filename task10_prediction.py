import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# File paths
DATA_FILE = Path("task10_raw_data.xlsx")
OUTPUT_DIR = Path("task10_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Load sessions
df = pd.read_excel(DATA_FILE, sheet_name="Sessions")

print("Original shape:", df.shape)
print("Columns:", df.columns.tolist())

# Identify target columns
targets = ["session_duration_minutes", "energy_kwh"]

# Basic cleaning
df = df.replace([np.inf, -np.inf], np.nan)

# Use only available, useful predictors.
# Avoid IDs and direct target leakage.
categorical_candidates = [
    "charger_type", "station_region", "user_type"
]
numeric_candidates = [
    "price_per_kwh", "year"
]

categorical = [c for c in categorical_candidates if c in df.columns]
numeric = [c for c in numeric_candidates if c in df.columns]
features = categorical + numeric

if not features:
    raise ValueError("No usable predictor columns were found.")

metrics = []
importance_tables = []

for target in targets:
    if target not in df.columns:
        print(f"Skipping missing target: {target}")
        continue

    data = df[features + [target]].copy()
    data[target] = pd.to_numeric(data[target], errors="coerce")
    data = data.dropna(subset=[target])

    # Remove missing predictor values
    for col in numeric:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(subset=features)

    if len(data) < 20:
        print(f"Not enough valid rows for {target}")
        continue

    X = data[features]
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    preprocessing = ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore"),
         categorical),
        ("numeric", "passthrough", numeric)
    ])

    model = Pipeline([
        ("preprocessing", preprocessing),
        ("model", RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
            min_samples_leaf=2
        ))
    ])

    print(f"\nTraining model for {target}...")
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    metrics.append({
        "target": target,
        "test_rows": len(y_test),
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

    # Feature importance
    encoder = model.named_steps["preprocessing"]
    transformed_names = encoder.get_feature_names_out()
    importances = model.named_steps["model"].feature_importances_

    importance_tables.append(pd.DataFrame({
        "target": target,
        "feature": transformed_names,
        "importance": importances
    }).sort_values("importance", ascending=False))

    print(f"MAE: {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R2: {r2:.3f}")

    # Save a sample of predictions
    pd.DataFrame({
        "actual": y_test.to_numpy(),
        "predicted": predictions
    }).to_csv(
        OUTPUT_DIR / f"{target}_predictions.csv", index=False
    )

# Save evaluation and feature importance
if metrics:
    pd.DataFrame(metrics).to_csv(
        OUTPUT_DIR / "model_metrics.csv", index=False
    )

if importance_tables:
    pd.concat(importance_tables, ignore_index=True).to_csv(
        OUTPUT_DIR / "feature_importance.csv", index=False
    )

print("\nFinished. Results saved in:", OUTPUT_DIR.resolve())

import matplotlib.pyplot as plt

# Chart 1: Actual vs Predicted values
for target in targets:
    file_path = OUTPUT_DIR / f"{target}_predictions.csv"

    if file_path.exists():
        results = pd.read_csv(file_path).head(200)

        plt.figure(figsize=(10, 5))
        plt.plot(results["actual"], label="Actual")
        plt.plot(results["predicted"], label="Predicted")
        plt.title(f"Actual vs Predicted - {target}")
        plt.xlabel("Test Sample")
        plt.ylabel(target)
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            OUTPUT_DIR / f"{target}_actual_vs_predicted.png",
            dpi=300
        )
        plt.close()

# Chart 2: Model performance
metrics_file = OUTPUT_DIR / "model_metrics.csv"

if metrics_file.exists():
    metrics_df = pd.read_csv(metrics_file)

    plt.figure(figsize=(8, 5))
    plt.bar(metrics_df["target"], metrics_df["R2"])
    plt.title("Model Performance - R2 Score")
    plt.xlabel("Prediction Target")
    plt.ylabel("R2 Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "model_performance_r2.png", dpi=300)
    plt.close()

# Chart 3: Feature importance
importance_file = OUTPUT_DIR / "feature_importance.csv"

if importance_file.exists():
    importance_df = pd.read_csv(importance_file)

    for target in importance_df["target"].unique():
        data = importance_df[importance_df["target"] == target]
        data = data.nlargest(10, "importance").sort_values("importance")

        plt.figure(figsize=(10, 6))
        plt.barh(data["feature"], data["importance"])
        plt.title(f"Top 10 Features - {target}")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.savefig(
            OUTPUT_DIR / f"{target}_feature_importance.png",
            dpi=300
        )
        plt.close()

print("All available charts saved successfully!")



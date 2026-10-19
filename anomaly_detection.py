from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# Project folders
project_folder = Path(__file__).resolve().parent
file_path = project_folder / "Data" / "ev_charging_dataset.xlsx"
results_folder = project_folder / "results"
results_folder.mkdir(exist_ok=True)

# Load Excel data
df = pd.read_excel(file_path, sheet_name="sessions")

# Normalize headings to avoid case and extra-space problems
df.columns = (
    df.columns.astype(str)
    .str.strip()
    .str.lower()
    .str.replace(" ", "_", regex=False)
)

print("Dataset loaded successfully!")
print("Rows:", len(df))
print("Columns:", df.columns.tolist())

# Match the column names after normalization
features = [
    "session_duration_minutes",
    "energy_kwh",
    "price_per_kwh",
    "total_cost"
]

missing = [col for col in features if col not in df.columns]

if missing:
    print("\nMissing columns:", missing)
    print("Please check the column names printed above.")

else:
    # Convert selected columns to numeric
    for col in features:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Keep valid charging sessions
    valid = df[features].notna().all(axis=1)
    valid &= df["session_duration_minutes"] > 0
    valid &= df["energy_kwh"] > 0
    valid &= df["price_per_kwh"] >= 0
    valid &= df["total_cost"] >= 0

    analysis = df.loc[valid].copy()

    if len(analysis) < 10:
        print("\nNot enough valid rows for anomaly detection.")

    else:
        model = IsolationForest(
            contamination=0.02,
            random_state=42
        )

        predictions = model.fit_predict(analysis[features])

        analysis["anomaly_status"] = [
            "Potential Anomaly" if p == -1 else "Normal"
            for p in predictions
        ]

        output_path = results_folder / "anomaly_results.xlsx"
        analysis.to_excel(output_path, index=False)

        print("\nAnalysis completed!")
        print("Rows analyzed:", len(analysis))
        print(
            "Potential anomalies:",
            (analysis["anomaly_status"] == "Potential Anomaly").sum()
        )
        print("Results saved to:", output_path)

# Create an anomaly summary chart
        counts = analysis["anomaly_status"].value_counts()

        counts = counts.reindex(
            ["Normal", "Potential Anomaly"],
            fill_value=0
        )

        plt.figure(figsize=(8, 5))
        counts.plot(kind="bar")
        plt.title("EV Charging Session Anomaly Detection")
        plt.xlabel("Session Status")
        plt.ylabel("Number of Sessions")
        plt.xticks(rotation=0)
        plt.tight_layout()

        chart_path = results_folder / "anomaly_summary.png"
        plt.savefig(chart_path, dpi=150)
        plt.close()

        print("Chart saved to:", chart_path)


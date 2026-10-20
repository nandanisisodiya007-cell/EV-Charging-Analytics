import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# File and output folder
file_path = "ev_charging_dataset.xlsx"
output_dir = Path("results")
output_dir.mkdir(exist_ok=True)

# Load session data
df = pd.read_excel(file_path, sheet_name="sessions")

# Clean important fields
df["customer_id"] = df["customer_id"].astype("string").str.strip()
df["total_cost"] = pd.to_numeric(df["total_cost"], errors="coerce")
df["energy_kwh"] = pd.to_numeric(df["energy_kwh"], errors="coerce")
df["start_timestamp"] = pd.to_datetime(
    df["start_timestamp"], errors="coerce"
)

df = df.dropna(subset=["customer_id", "total_cost"])
df = df[df["customer_id"] != ""]

# Summarize each customer
clv = df.groupby("customer_id").agg(
    total_sessions=("session_id", "nunique"),
    total_spend=("total_cost", "sum"),
    total_energy_kwh=("energy_kwh", "sum"),
    first_session=("start_timestamp", "min"),
    last_session=("start_timestamp", "max")
).reset_index()

# Calculate customer activity and estimated value
clv["average_spend_per_session"] = (
    clv["total_spend"] / clv["total_sessions"]
)

clv["active_days"] = (
    clv["last_session"] - clv["first_session"]
).dt.days + 1

clv["estimated_annual_value"] = (
    clv["average_spend_per_session"]
    * clv["total_sessions"]
    / (clv["active_days"] / 365).clip(lower=1/365)
)

# Rank customers into value segments
clv["value_segment"] = pd.qcut(
    clv["total_spend"].rank(method="first"),
    q=3,
    labels=["Low Value", "Medium Value", "High Value"]
)

# Save results
clv = clv.sort_values("total_spend", ascending=False)
clv.to_csv(output_dir / "customer_clv_results.csv", index=False)
clv.to_excel(output_dir / "customer_clv_results.xlsx", index=False)

# Create customer value chart
segment_spend = clv.groupby(
    "value_segment", observed=False
)["total_spend"].sum()

plt.figure(figsize=(8, 5))
segment_spend.plot(kind="bar")
plt.title("Total Customer Spend by Value Segment")
plt.xlabel("Customer Segment")
plt.ylabel("Total Spend")
plt.tight_layout()
plt.savefig(output_dir / "clv_segment_chart.png")
plt.close()

print("CLV analysis completed successfully!")
print("Customers analyzed:", len(clv))
print("\nCustomer segments:")
print(clv["value_segment"].value_counts())
print("\nTop 10 customers:")
print(clv.head(10).to_string(index=False))
print("\nResults saved in the results folder.")


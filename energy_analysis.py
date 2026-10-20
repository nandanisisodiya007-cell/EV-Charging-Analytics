import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# File paths
file_path = "ev_charging_dataset.xlsx"
output_dir = Path("results")
output_dir.mkdir(exist_ok=True)

print("Loading charging session data...")
df = pd.read_excel(file_path, sheet_name="sessions")

# Clean and prepare data
df["start_timestamp"] = pd.to_datetime(
    df["start_timestamp"], errors="coerce"
)
df["energy_kwh"] = pd.to_numeric(
    df["energy_kwh"], errors="coerce"
)
df["station_id"] = df["station_id"].astype("string")
df["station_region"] = df["station_region"].astype("string")

df = df.dropna(
    subset=["start_timestamp", "energy_kwh", "station_id"]
)
df = df[df["energy_kwh"] >= 0].copy()

# Add time fields
df["date"] = df["start_timestamp"].dt.date
df["hour"] = df["start_timestamp"].dt.hour
df["month"] = df["start_timestamp"].dt.to_period("M").astype(str)

# 1. Total energy by station
station_energy = df.groupby("station_id").agg(
    total_kwh=("energy_kwh", "sum"),
    total_sessions=("session_id", "nunique")
).reset_index()

station_energy.to_csv(
    output_dir / "station_energy_summary.csv", index=False
)

# 2. Total energy by region
region_energy = df.groupby("station_region").agg(
    total_kwh=("energy_kwh", "sum"),
    total_sessions=("session_id", "nunique")
).reset_index().sort_values("total_kwh", ascending=False)

region_energy.to_csv(
    output_dir / "regional_energy_summary.csv", index=False
)

# 3. Peak demand periods by hour
hourly_energy = df.groupby("hour").agg(
    total_kwh=("energy_kwh", "sum"),
    total_sessions=("session_id", "nunique")
).reset_index()

hourly_energy.to_csv(
    output_dir / "hourly_load_summary.csv", index=False
)

# 4. Daily energy demand
daily_energy = df.groupby("date").agg(
    total_kwh=("energy_kwh", "sum"),
    total_sessions=("session_id", "nunique")
).reset_index()

daily_energy["date"] = pd.to_datetime(daily_energy["date"])
daily_energy = daily_energy.sort_values("date")
daily_energy.to_csv(
    output_dir / "daily_energy_summary.csv", index=False
)

# 5. Basic daily energy forecast
daily_energy["forecast_kwh"] = (
    daily_energy["total_kwh"].rolling(
        window=7, min_periods=1
    ).mean()
)

daily_energy.to_csv(
    output_dir / "daily_energy_forecast.csv", index=False
)

# Chart 1: Energy by region
plt.figure(figsize=(9, 5))
region_energy.plot(
    x="station_region", y="total_kwh",
    kind="bar", legend=False
)
plt.title("Total Energy Consumption by Region")
plt.xlabel("Region")
plt.ylabel("Energy (kWh)")
plt.tight_layout()
plt.savefig(output_dir / "regional_energy_chart.png")
plt.close()

# Chart 2: Energy by hour
plt.figure(figsize=(9, 5))
plt.plot(
    hourly_energy["hour"],
    hourly_energy["total_kwh"],
    marker="o"
)
plt.title("Energy Consumption by Hour of Day")
plt.xlabel("Hour of Day")
plt.ylabel("Energy (kWh)")
plt.xticks(range(0, 24))
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / "hourly_load_chart.png")
plt.close()

# Chart 3: Daily energy and rolling-average estimate
plt.figure(figsize=(10, 5))
plt.plot(
    daily_energy["date"],
    daily_energy["total_kwh"],
    label="Observed daily energy"
)
plt.plot(
    daily_energy["date"],
    daily_energy["forecast_kwh"],
    label="7-day rolling average"
)
plt.title("Daily Energy Consumption and Rolling Average")
plt.xlabel("Date")
plt.ylabel("Energy (kWh)")
plt.legend()
plt.tight_layout()
plt.savefig(output_dir / "daily_energy_forecast_chart.png")
plt.close()

# Summary report
summary = pd.DataFrame([{
    "total_sessions": df["session_id"].nunique(),
    "total_energy_kwh": df["energy_kwh"].sum(),
    "station_count": df["station_id"].nunique(),
    "region_count": df["station_region"].nunique(),
    "peak_hour": int(
        hourly_energy.loc[
            hourly_energy["total_kwh"].idxmax(), "hour"
        ]
    ),
    "peak_hour_energy_kwh": hourly_energy["total_kwh"].max()
}])

summary.to_csv(output_dir / "overall_energy_summary.csv", index=False)

print("\nENERGY ANALYSIS COMPLETED!")
print("\nOverall summary:")
print(summary.to_string(index=False))
print("\nTop 5 stations by energy:")
print(station_energy.nlargest(5, "total_kwh").to_string(index=False))
print("\nRegional energy summary:")
print(region_energy.to_string(index=False))
print(f"\nAll results saved in: {output_dir.resolve()}")


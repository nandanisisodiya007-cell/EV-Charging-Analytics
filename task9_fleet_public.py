import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# TASK 9 - FLEET VS PUBLIC USER BEHAVIOUR ANALYSIS
# ============================================================

# ---------------- FILE SETTINGS ----------------
INPUT_FILE = "task_9.xlsx"
OUTPUT_FILE = "task9_python_results.xlsx"

# ---------------- LOAD DATA ----------------
print("Loading dataset...")

sessions = pd.read_excel(
    INPUT_FILE,
    sheet_name="Sessions"
)

print("Dataset loaded successfully.")
print(f"Rows: {len(sessions):,}")
print(f"Columns: {len(sessions.columns)}")

# ---------------- CLEAN COLUMN NAMES ----------------
sessions.columns = (
    sessions.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)

print("\nColumns found:")
print(list(sessions.columns))

# ---------------- CHECK REQUIRED COLUMNS ----------------
required_columns = [
    "session_id",
    "station_id",
    "charger_id",
    "charger_type",
    "start_timestamp",
    "end_timestamp",
    "session_duration_minutes",
    "energy_kwh",
    "price_per_kwh",
    "total_cost",
    "payment_method",
    "user_type",
    "customer_id",
    "station_region",
    "year"
]

missing_columns = [
    col for col in required_columns
    if col not in sessions.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns: {missing_columns}"
    )

# ---------------- CLEAN DATA ----------------
sessions["energy_kwh"] = pd.to_numeric(
    sessions["energy_kwh"],
    errors="coerce"
)

sessions["total_cost"] = pd.to_numeric(
    sessions["total_cost"],
    errors="coerce"
)

sessions["session_duration_minutes"] = pd.to_numeric(
    sessions["session_duration_minutes"],
    errors="coerce"
)

sessions["price_per_kwh"] = pd.to_numeric(
    sessions["price_per_kwh"],
    errors="coerce"
)

sessions["user_type"] = (
    sessions["user_type"]
    .astype(str)
    .str.strip()
    .str.lower()
)

sessions["station_region"] = (
    sessions["station_region"]
    .astype(str)
    .str.strip()
)

sessions["charger_type"] = (
    sessions["charger_type"]
    .astype(str)
    .str.strip()
)

# Remove rows with essential missing values
sessions = sessions.dropna(
    subset=[
        "user_type",
        "energy_kwh",
        "total_cost"
    ]
)

print(f"Clean rows: {len(sessions):,}")

# ============================================================
# 1. USER TYPE ANALYSIS
# ============================================================

user_type_analysis = (
    sessions
    .groupby("user_type")
    .agg(
        Sessions=("session_id", "count"),
        Total_Energy_kWh=("energy_kwh", "sum"),
        Total_Revenue=("total_cost", "sum"),
        Avg_Energy_per_Session=("energy_kwh", "mean"),
        Avg_Revenue_per_Session=("total_cost", "mean")
    )
    .reset_index()
)

# ============================================================
# 2. FLEET VS PUBLIC COMPARISON
# ============================================================

fleet_public = sessions[
    sessions["user_type"].isin(["fleet", "public"])
].copy()

fleet_public_comparison = (
    fleet_public
    .groupby("user_type")
    .agg(
        Sessions=("session_id", "count"),
        Total_Energy_kWh=("energy_kwh", "sum"),
        Total_Revenue=("total_cost", "sum"),
        Avg_Energy_per_Session=("energy_kwh", "mean"),
        Avg_Revenue_per_Session=("total_cost", "mean")
    )
    .reset_index()
)

# Revenue share
total_fp_revenue = fleet_public_comparison[
    "Total_Revenue"
].sum()

fleet_public_comparison["Revenue_Share_%"] = (
    fleet_public_comparison["Total_Revenue"]
    / total_fp_revenue * 100
)

# ============================================================
# 3. BEHAVIOUR BY REGION
# ============================================================

region_analysis = (
    sessions
    .groupby("station_region")
    .agg(
        Sessions=("session_id", "count"),
        Total_Energy_kWh=("energy_kwh", "sum"),
        Total_Revenue=("total_cost", "sum"),
        Avg_Energy_per_Session=("energy_kwh", "mean"),
        Avg_Revenue_per_Session=("total_cost", "mean")
    )
    .reset_index()
    .sort_values("Sessions", ascending=False)
)

# ============================================================
# 4. BEHAVIOUR BY CHARGER TYPE
# ============================================================

charger_analysis = (
    sessions
    .groupby("charger_type")
    .agg(
        Sessions=("session_id", "count"),
        Total_Energy_kWh=("energy_kwh", "sum"),
        Total_Revenue=("total_cost", "sum"),
        Avg_Energy_per_Session=("energy_kwh", "mean"),
        Avg_Revenue_per_Session=("total_cost", "mean")
    )
    .reset_index()
    .sort_values("Sessions", ascending=False)
)

# ============================================================
# 5. USER TYPE + REGION ANALYSIS
# ============================================================

user_region_analysis = (
    sessions
    .groupby(["user_type", "station_region"])
    .agg(
        Sessions=("session_id", "count"),
        Total_Energy_kWh=("energy_kwh", "sum"),
        Total_Revenue=("total_cost", "sum")
    )
    .reset_index()
)

# ============================================================
# 6. USER TYPE + CHARGER ANALYSIS
# ============================================================

user_charger_analysis = (
    sessions
    .groupby(["user_type", "charger_type"])
    .agg(
        Sessions=("session_id", "count"),
        Total_Energy_kWh=("energy_kwh", "sum"),
        Total_Revenue=("total_cost", "sum")
    )
    .reset_index()
)

# ============================================================
# 7. KEY INSIGHTS
# ============================================================

insights = []

# Fleet vs public revenue
if len(fleet_public_comparison) > 0:

    fleet_data = fleet_public_comparison[
        fleet_public_comparison["user_type"] == "fleet"
    ]

    public_data = fleet_public_comparison[
        fleet_public_comparison["user_type"] == "public"
    ]

    if not fleet_data.empty and not public_data.empty:

        fleet_rev = fleet_data.iloc[0]["Total_Revenue"]
        public_rev = public_data.iloc[0]["Total_Revenue"]

        fleet_energy = fleet_data.iloc[0]["Avg_Energy_per_Session"]
        public_energy = public_data.iloc[0]["Avg_Energy_per_Session"]

        if fleet_rev > public_rev:
            insights.append(
                "Fleet users generate higher total charging revenue than public users."
            )
        else:
            insights.append(
                "Public users generate higher total charging revenue than fleet users."
            )

        if fleet_energy > public_energy:
            insights.append(
                "Fleet users have higher average energy consumption per session than public users."
            )
        else:
            insights.append(
                "Public users have higher average energy consumption per session than fleet users."
            )

        insights.append(
            "Fleet and public users can both be important customer segments for the charging network."
        )

# Highest session region
if not region_analysis.empty:
    top_region = region_analysis.iloc[0]["station_region"]

    insights.append(
        f"{top_region} has the highest number of charging sessions."
    )

# Highest charger type
if not charger_analysis.empty:
    top_charger = charger_analysis.iloc[0]["charger_type"]

    insights.append(
        f"{top_charger} has the highest charging session volume."
    )

insights_df = pd.DataFrame({
    "Key Behaviour Insights": insights
})

# ============================================================
# 8. FLEET PARTNERSHIP RECOMMENDATIONS
# ============================================================

recommendations = [
    "Offer dedicated fleet charging plans for high-frequency users.",
    "Provide discounted or volume-based pricing for fleet partners.",
    "Reserve selected charging capacity during high-demand periods for contracted fleets.",
    "Use fleet charging patterns to plan station capacity and staffing.",
    "Develop long-term partnerships with taxi and delivery fleets."
]

recommendations_df = pd.DataFrame({
    "Fleet Partnership Recommendations": recommendations
})

# ============================================================
# 9. CREATE OUTPUT EXCEL FILE
# ============================================================

print("\nCreating Excel output...")

with pd.ExcelWriter(
    OUTPUT_FILE,
    engine="openpyxl"
) as writer:

    user_type_analysis.to_excel(
        writer,
        sheet_name="User Type Analysis",
        index=False
    )

    fleet_public_comparison.to_excel(
        writer,
        sheet_name="Fleet vs Public",
        index=False
    )

    region_analysis.to_excel(
        writer,
        sheet_name="Behaviour by Region",
        index=False
    )

    charger_analysis.to_excel(
        writer,
        sheet_name="Behaviour by Charger",
        index=False
    )

    user_region_analysis.to_excel(
        writer,
        sheet_name="User Region Analysis",
        index=False
    )

    user_charger_analysis.to_excel(
        writer,
        sheet_name="User Charger Analysis",
        index=False
    )

    insights_df.to_excel(
        writer,
        sheet_name="Key Insights",
        index=False
    )

    recommendations_df.to_excel(
        writer,
        sheet_name="Recommendations",
        index=False
    )

# ============================================================
# 10. CREATE CHARTS
# ============================================================

print("Creating charts...")

# Chart 1 - Charging Sessions by User Type
plt.figure(figsize=(8, 5))

plt.bar(
    user_type_analysis["user_type"],
    user_type_analysis["Sessions"]
)

plt.title("Charging Sessions by User Type")
plt.xlabel("User Type")
plt.ylabel("Number of Sessions")
plt.xticks(rotation=30)
plt.tight_layout()

plt.savefig(
    "task9_sessions_by_user_type.png",
    dpi=300
)

plt.close()

# Chart 2 - Energy Consumption by User Type
plt.figure(figsize=(8, 5))

plt.bar(
    user_type_analysis["user_type"],
    user_type_analysis["Total_Energy_kWh"]
)

plt.title("Energy Consumption by User Type")
plt.xlabel("User Type")
plt.ylabel("Total Energy (kWh)")
plt.xticks(rotation=30)
plt.tight_layout()

plt.savefig(
    "task9_energy_by_user_type.png",
    dpi=300
)

plt.close()

# Chart 3 - Revenue by User Type
plt.figure(figsize=(8, 5))

plt.bar(
    user_type_analysis["user_type"],
    user_type_analysis["Total_Revenue"]
)

plt.title("Revenue Contribution by User Type")
plt.xlabel("User Type")
plt.ylabel("Total Revenue")
plt.xticks(rotation=30)
plt.tight_layout()

plt.savefig(
    "task9_revenue_by_user_type.png",
    dpi=300
)

plt.close()

# Chart 4 - Sessions by Region
plt.figure(figsize=(10, 6))

plt.bar(
    region_analysis["station_region"],
    region_analysis["Sessions"]
)

plt.title("Charging Sessions by Region")
plt.xlabel("Region")
plt.ylabel("Number of Sessions")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

plt.savefig(
    "task9_sessions_by_region.png",
    dpi=300
)

plt.close()

# Chart 5 - Sessions by Charger Type
plt.figure(figsize=(8, 5))

plt.bar(
    charger_analysis["charger_type"],
    charger_analysis["Sessions"]
)

plt.title("Charging Sessions by Charger Type")
plt.xlabel("Charger Type")
plt.ylabel("Number of Sessions")
plt.xticks(rotation=30)
plt.tight_layout()

plt.savefig(
    "task9_sessions_by_charger.png",
    dpi=300
)

plt.close()

print("\n========================================")
print("TASK 9 COMPLETED SUCCESSFULLY")
print("========================================")
print(f"Output Excel: {OUTPUT_FILE}")
print("Charts created:")
print("1. task9_sessions_by_user_type.png")
print("2. task9_energy_by_user_type.png")
print("3. task9_revenue_by_user_type.png")
print("4. task9_sessions_by_region.png")
print("5. task9_sessions_by_charger.png")
print("========================================")

import pandas as pd
from pathlib import Path

# ==========================================
# TASK 13: STATION PERFORMANCE RANKING SYSTEM
# ==========================================

# 1. File paths
folder = Path(__file__).resolve().parent
input_file = folder / "Task 13.xlsx"
output_file = folder / "Task_13_Python_Results.xlsx"

# 2. Check the Excel workbook
if not input_file.exists():
    raise FileNotFoundError(
        f"Excel workbook not found: {input_file}\n"
        "Place Task_13_Station_Performance.xlsx "
        "in the same folder as this Python file."
    )

# 3. Load the station performance sheet
excel = pd.ExcelFile(input_file)

sheet_name = next(
    (
        sheet for sheet in excel.sheet_names
        if sheet.strip().lower() == "station performance"
    ),
    None
)

if sheet_name is None:
    raise ValueError(
        "Sheet 'Station performance' was not found. "
        f"Available sheets: {excel.sheet_names}"
    )

df = pd.read_excel(input_file, sheet_name=sheet_name)
df.columns = df.columns.astype(str).str.strip()

# 4. Identify columns
def find_column(possible_names):
    for col in df.columns:
        normalized = col.strip().lower().replace("_", " ")
        if normalized in possible_names:
            return col
    return None

station_col = find_column({
    "station id", "station"
})
region_col = find_column({"region"})
score_col = find_column({
    "performance score", "performance scores",
    "score", "overall performance score"
})
category_col = find_column({
    "performance category", "category"
})
rank_col = find_column({
    "performance rank", "station rank", "rank"
})
sessions_col = find_column({
    "total sessions", "total session", "sessions"
})
energy_col = find_column({
    "total energy (kwh)", "total energy",
    "energy delivered", "energy"
})
revenue_col = find_column({
    "total revenue", "revenue"
})
customers_col = find_column({
    "unique customers", "total customers",
    "unique customer"
})
recommendation_col = find_column({
    "recommendation", "recommendation action"
})

if station_col is None or score_col is None:
    raise ValueError(
        "Station ID or Performance Score column not found.\n"
        f"Your actual column names are: {df.columns.tolist()}\n"
        "Update the column names in the script if necessary."
    )

# 5. Clean and validate the data
df[score_col] = pd.to_numeric(df[score_col], errors="coerce")

df = df.dropna(subset=[station_col, score_col]).copy()
df[station_col] = df[station_col].astype(str).str.strip()
df = df[df[station_col] != ""]

if df[station_col].duplicated().any():
    raise ValueError(
        "Duplicate station IDs found. "
        "Please check the Station performance sheet."
    )

if df.empty:
    raise ValueError("No valid station records were found.")

# 6. Calculate station ranking from performance scores
df["Python Performance Rank"] = (
    df[score_col].rank(method="min", ascending=False).astype(int)
)

df = df.sort_values(
    [score_col, station_col],
    ascending=[False, True]
).reset_index(drop=True)

# 7. Create the top 5 station table
top5 = df.head(5).copy()

# 8. Identify underperforming stations
if category_col is not None:
    category_text = df[category_col].astype(str).str.strip().str.lower()
    needs_improvement = category_text.str.contains(
        "needs improvement|underperform", regex=True, na=False
    )
    underperforming = df[needs_improvement].copy()
else:
    # If no category exists, use the five lowest-scoring stations
    underperforming = df.nsmallest(5, score_col).copy()

# 9. Create a KPI comparison table
kpi_columns = [
    col for col in [
        station_col, region_col, sessions_col,
        energy_col, revenue_col, customers_col,
        score_col, "Python Performance Rank"
    ]
    if col is not None
]

kpi_comparison = df[kpi_columns].copy()

# 10. Check ranking against the existing Excel rank, if available
validation = pd.DataFrame({
    "Station ID": df[station_col],
    "Excel Performance Score": df[score_col],
    "Python Performance Rank": df["Python Performance Rank"]
})

if rank_col is not None:
    excel_ranks = pd.to_numeric(df[rank_col], errors="coerce")
    validation["Excel Performance Rank"] = excel_ranks
    validation["Rank Match"] = (
        excel_ranks == df["Python Performance Rank"]
    )

# 11. Create a summary report
summary = pd.DataFrame({
    "Metric": [
        "Total stations analysed",
        "Average performance score",
        "Highest performance score",
        "Lowest performance score",
        "Underperforming stations identified"
    ],
    "Value": [
        len(df),
        round(df[score_col].mean(), 2),
        round(df[score_col].max(), 2),
        round(df[score_col].min(), 2),
        len(underperforming)
    ]
})

# 12. Export results into a new workbook
# The original Excel workbook is not modified.
with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
    summary.to_excel(
        writer, sheet_name="Summary", index=False
    )
    kpi_comparison.to_excel(
        writer, sheet_name="KPI Comparison", index=False
    )
    df.to_excel(
        writer, sheet_name="Station Rankings", index=False
    )
    top5.to_excel(
        writer, sheet_name="Top 5 Stations", index=False
    )
    underperforming.to_excel(
        writer, sheet_name="Underperforming", index=False
    )
    validation.to_excel(
        writer, sheet_name="Ranking Validation", index=False
    )

print("\nTASK 13 PYTHON ANALYSIS COMPLETED")
print("---------------------------------")
print(f"Stations analysed: {len(df)}")
print(f"Top 5 stations:\n{top5[[station_col, score_col]]}")
print(f"\nUnderperforming stations: {len(underperforming)}")
print(f"\nResults saved to: {output_file}")
print("\nOriginal Excel workbook remains unchanged.")


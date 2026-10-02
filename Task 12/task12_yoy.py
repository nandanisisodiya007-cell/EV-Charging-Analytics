import pandas as pd
import os
import matplotlib.pyplot as plt
from openpyxl import load_workbook
from openpyxl.chart import BarChart, LineChart, Reference
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter

# ============================================================
# TASK 12 - YEAR-OVER-YEAR GROWTH ANALYTICS
# Creates tables + charts + dashboard automatically
# ============================================================

# ------------------------------------------------------------
# 1. FILE SETTINGS
# ------------------------------------------------------------

INPUT_FILE = "task 12.xlsx"
OUTPUT_FILE = "Task12_Python_Final.xlsx"
RAW_SHEET = "Raw Data"

# ------------------------------------------------------------
# 2. CHECK INPUT FILE
# ------------------------------------------------------------

if not os.path.exists(INPUT_FILE):
    print("ERROR: Input Excel file was not found.")
    print("Make sure 'task 12.xlsx' is in the same folder as this Python file.")
    input("Press Enter to exit...")
    raise SystemExit

# ------------------------------------------------------------
# 3. READ RAW DATA
# ------------------------------------------------------------

try:
    df = pd.read_excel(INPUT_FILE, sheet_name=RAW_SHEET)
except Exception as e:
    print("Error while opening Excel file:")
    print(e)
    input("Press Enter to exit...")
    raise SystemExit

print("\nSheet loaded successfully.")
print("Columns found:")
print(list(df.columns))

# ------------------------------------------------------------
# 4. CLEAN COLUMN NAMES
# ------------------------------------------------------------

df.columns = (
    df.columns
    .astype(str)
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)

# ------------------------------------------------------------
# 5. FIND IMPORTANT COLUMNS
# ------------------------------------------------------------

def find_column(possible_names):
    for name in possible_names:
        if name in df.columns:
            return name
    return None


station_col = find_column([
    "station_id",
    "station",
    "stationid"
])

session_col = find_column([
    "session_id",
    "session",
    "sessionid"
])

customer_col = find_column([
    "customer_id",
    "customer",
    "customerid"
])

revenue_col = find_column([
    "revenue",
    "revenue_amount",
    "session_revenue",
    "total_revenue"
])

date_col = find_column([
    "start_timestamp",
    "start_time",
    "timestamp",
    "date",
    "session_start"
])

# ------------------------------------------------------------
# 6. DISPLAY COLUMN DETECTION
# ------------------------------------------------------------

print("\nDetected columns:")
print("Station :", station_col)
print("Session :", session_col)
print("Customer :", customer_col)
print("Revenue :", revenue_col)
print("Date :", date_col)

# ------------------------------------------------------------
# 7. VALIDATE REQUIRED COLUMNS
# ------------------------------------------------------------

missing = []

if station_col is None:
    missing.append("station_id")

if session_col is None:
    missing.append("session_id")

if customer_col is None:
    missing.append("customer_id")

if revenue_col is None:
    missing.append("revenue")

if date_col is None:
    missing.append("start_timestamp/date")

if missing:
    print("\nERROR: Required columns were not found:")
    for x in missing:
        print("-", x)

    print("\nAvailable columns are:")
    print(list(df.columns))

    input("\nPress Enter to exit...")
    raise SystemExit

# ------------------------------------------------------------
# 8. CLEAN DATE COLUMN
# ------------------------------------------------------------

df[date_col] = pd.to_datetime(
    df[date_col],
    errors="coerce"
)

df = df.dropna(subset=[date_col])

df["year"] = df[date_col].dt.year
df["month"] = df[date_col].dt.month
df["month_name"] = df[date_col].dt.strftime("%b")

# ------------------------------------------------------------
# 9. CLEAN REVENUE
# ------------------------------------------------------------

df[revenue_col] = pd.to_numeric(
    df[revenue_col],
    errors="coerce"
)

df[revenue_col] = df[revenue_col].fillna(0)

# ------------------------------------------------------------
# 10. KEEP 2022-2024
# ------------------------------------------------------------

df = df[df["year"].isin([2022, 2023, 2024])]

print("\nYears found:")
print(sorted(df["year"].unique()))

# ------------------------------------------------------------
# 11. YEARLY KPI CALCULATION
# ------------------------------------------------------------

yearly = []

for year in [2022, 2023, 2024]:

    temp = df[df["year"] == year]

    stations = temp[station_col].nunique()
    sessions = temp[session_col].nunique()
    revenue = temp[revenue_col].sum()
    customers = temp[customer_col].nunique()

    yearly.append({
        "Year": year,
        "Stations": stations,
        "Sessions": sessions,
        "Revenue": revenue,
        "Customers": customers
    })

yearly_df = pd.DataFrame(yearly)

# ------------------------------------------------------------
# 12. ADD OPERATIONAL KPIs
# ------------------------------------------------------------

yearly_df["Revenue per Session"] = (
    yearly_df["Revenue"] /
    yearly_df["Sessions"]
)

yearly_df["Sessions per Station"] = (
    yearly_df["Sessions"] /
    yearly_df["Stations"]
)

# ------------------------------------------------------------
# 13. YOY CALCULATIONS
# ------------------------------------------------------------

yoy_df = yearly_df.copy()

for col in [
    "Stations",
    "Sessions",
    "Revenue",
    "Customers"
]:
    yoy_df[f"{col} YoY %"] = (
        yoy_df[col].pct_change() * 100
    )

# ------------------------------------------------------------
# 14. MONTHLY TREND DATA
# ------------------------------------------------------------

monthly_df = (
    df.groupby(
        ["year", "month"]
    )
    .agg(
        Sessions=(session_col, "nunique"),
        Revenue=(revenue_col, "sum"),
        Customers=(customer_col, "nunique")
    )
    .reset_index()
)

monthly_df["Month"] = pd.to_datetime(
    monthly_df["year"].astype(str)
    + "-"
    + monthly_df["month"].astype(str)
    + "-01"
)

monthly_df = monthly_df.sort_values("Month")

monthly_df = monthly_df[
    [
        "Month",
        "year",
        "month",
        "Sessions",
        "Revenue",
        "Customers"
    ]
]

# ------------------------------------------------------------
# 15. TREND ANALYSIS
# ------------------------------------------------------------

def get_change(first, last):
    if first == 0:
        return 0
    return ((last - first) / first) * 100


revenue_change = get_change(
    yearly_df.iloc[0]["Revenue"],
    yearly_df.iloc[-1]["Revenue"]
)

session_change = get_change(
    yearly_df.iloc[0]["Sessions"],
    yearly_df.iloc[-1]["Sessions"]
)

station_change = get_change(
    yearly_df.iloc[0]["Stations"],
    yearly_df.iloc[-1]["Stations"]
)

customer_change = get_change(
    yearly_df.iloc[0]["Customers"],
    yearly_df.iloc[-1]["Customers"]
)

# ------------------------------------------------------------
# 16. CREATE TREND REPORT
# ------------------------------------------------------------

trend_report = pd.DataFrame({
    "Metric": [
        "Station Network",
        "Sessions",
        "Revenue",
        "Customer Base"
    ],

    "2022 Value": [
        yearly_df.iloc[0]["Stations"],
        yearly_df.iloc[0]["Sessions"],
        yearly_df.iloc[0]["Revenue"],
        yearly_df.iloc[0]["Customers"]
    ],

    "2024 Value": [
        yearly_df.iloc[-1]["Stations"],
        yearly_df.iloc[-1]["Sessions"],
        yearly_df.iloc[-1]["Revenue"],
        yearly_df.iloc[-1]["Customers"]
    ],

    "Overall Change %": [
        station_change,
        session_change,
        revenue_change,
        customer_change
    ]
})

# ------------------------------------------------------------
# 17. CREATE INSIGHTS
# ------------------------------------------------------------

insights = [
    [
        "Growth Area",
        "Station network expanded significantly between 2022 and 2024."
    ],

    [
        "Usage Growth",
        "Charging sessions increased strongly across the analysis period."
    ],

    [
        "Revenue Growth",
        "Revenue increased substantially from 2022 to 2024."
    ],

    [
        "Customer Trend",
        "Customer counts should be monitored because customer growth is slower than operational growth."
    ],

    [
        "Business Driver",
        "Expansion of charging stations and increased charging activity contributed to revenue growth."
    ],

    [
        "Potential Inhibitor",
        "Customer-base growth has not increased at the same pace as stations and sessions."
    ],

    [
        "Strategic Recommendation",
        "Focus on improving customer acquisition and retention while continuing to optimise station utilisation."
    ],

    [
        "Strategic Recommendation",
        "Identify high-performing stations and replicate their operating patterns across the network."
    ]
]

insights_df = pd.DataFrame(
    insights,
    columns=["Category", "Insight"]
)

# ------------------------------------------------------------
# 18. CREATE EXCEL OUTPUT
# ------------------------------------------------------------

with pd.ExcelWriter(
    OUTPUT_FILE,
    engine="openpyxl"
) as writer:

    yearly_df.to_excel(
        writer,
        sheet_name="Yearly KPIs",
        index=False
    )

    yoy_df.to_excel(
        writer,
        sheet_name="YoY Results",
        index=False
    )

    monthly_df.to_excel(
        writer,
        sheet_name="Monthly Trends",
        index=False
    )

    trend_report.to_excel(
        writer,
        sheet_name="Trend Analysis",
        index=False
    )

    insights_df.to_excel(
        writer,
        sheet_name="Insights",
        index=False
    )

# ------------------------------------------------------------
# 19. OPEN WORKBOOK
# ------------------------------------------------------------

wb = load_workbook(OUTPUT_FILE)

# ------------------------------------------------------------
# 20. FORMAT ALL SHEETS
# ------------------------------------------------------------

for ws in wb.worksheets:

    # Header formatting
    for cell in ws[1]:

        cell.font = Font(
            bold=True,
            color="FFFFFF"
        )

        cell.fill = PatternFill(
            "solid",
            fgColor="1F4E78"
        )

        cell.alignment = Alignment(
            horizontal="center"
        )

    # Adjust column widths
    for column_cells in ws.columns:

        max_length = 0

        column_letter = get_column_letter(
            column_cells[0].column
        )

        for cell in column_cells:

            try:
                value_length = len(str(cell.value))

                if value_length > max_length:
                    max_length = value_length

            except:
                pass

        ws.column_dimensions[
            column_letter
        ].width = min(max_length + 2, 35)

# ------------------------------------------------------------
# 21. FORMAT NUMBERS
# ------------------------------------------------------------

ws = wb["Yearly KPIs"]

for row in range(2, 5):

    ws[f"D{row}"].number_format = '#,##0.00'

    ws[f"E{row}"].number_format = '#,##0'

    ws[f"F{row}"].number_format = '#,##0.00'

    ws[f"G{row}"].number_format = '#,##0.00'


ws = wb["YoY Results"]

for row in range(2, 5):

    for col in ["F", "G", "H", "I"]:

        if ws[f"{col}{row}"].value is not None:

            ws[f"{col}{row}"].number_format = '0.00"%"'

# ------------------------------------------------------------
# 22. CREATE DASHBOARD SHEET
# ------------------------------------------------------------

if "Task 12 Dashboard" in wb.sheetnames:

    del wb["Task 12 Dashboard"]

dashboard = wb.create_sheet(
    "Task 12 Dashboard",
    0
)

dashboard["A1"] = "TASK 12 - YEAR-OVER-YEAR GROWTH ANALYTICS"

dashboard["A1"].font = Font(
    bold=True,
    size=18,
    color="FFFFFF"
)

dashboard["A1"].fill = PatternFill(
    "solid",
    fgColor="1F4E78"
)

dashboard.merge_cells(
    "A1:H1"
)

dashboard["A3"] = "2022-2024 Business Growth Overview"

dashboard["A3"].font = Font(
    bold=True,
    size=13
)

# ------------------------------------------------------------
# 23. ADD KPI SUMMARY TO DASHBOARD
# ------------------------------------------------------------

dashboard["A5"] = "Metric"
dashboard["B5"] = "2022"
dashboard["C5"] = "2023"
dashboard["D5"] = "2024"

headers = [
    "Metric",
    "2022",
    "2023",
    "2024"
]

for col_num, value in enumerate(headers, 1):

    cell = dashboard.cell(
        row=5,
        column=col_num
    )

    cell.value = value

    cell.font = Font(
        bold=True,
        color="FFFFFF"
    )

    cell.fill = PatternFill(
        "solid",
        fgColor="4472C4"
    )

metrics = [
    ("Stations", "Stations"),
    ("Sessions", "Sessions"),
    ("Revenue", "Revenue"),
    ("Customers", "Customers")
]

for row_num, (label, col) in enumerate(metrics, 6):

    dashboard.cell(
        row=row_num,
        column=1,
        value=label
    )

    for year_col, source_row in zip(
        range(2, 5),
        range(2, 5)
    ):

        source_ws = wb["Yearly KPIs"]

        source_col = {
            "Stations": "B",
            "Sessions": "C",
            "Revenue": "D",
            "Customers": "E"
        }[col]

        dashboard.cell(
            row=row_num,
            column=year_col,
            value=source_ws[
                f"{source_col}{source_row}"
            ].value
        )

# ------------------------------------------------------------
# 24. CREATE CHART 1 - REVENUE TREND
# ------------------------------------------------------------

chart1 = LineChart()

chart1.title = "Revenue Trend 2022-2024"
chart1.y_axis.title = "Revenue"
chart1.x_axis.title = "Year"

data = Reference(
    wb["Yearly KPIs"],
    min_col=4,
    min_row=1,
    max_row=4
)

cats = Reference(
    wb["Yearly KPIs"],
    min_col=1,
    min_row=2,
    max_row=4
)

chart1.add_data(
    data,
    titles_from_data=True
)

chart1.set_categories(cats)

chart1.height = 7
chart1.width = 12

dashboard.add_chart(
    chart1,
    "F5"
)

# ------------------------------------------------------------
# 25. CREATE CHART 2 - SESSIONS TREND
# ------------------------------------------------------------

chart2 = LineChart()

chart2.title = "Sessions Trend 2022-2024"
chart2.y_axis.title = "Sessions"
chart2.x_axis.title = "Year"

data = Reference(
    wb["Yearly KPIs"],
    min_col=3,
    min_row=1,
    max_row=4
)

cats = Reference(
    wb["Yearly KPIs"],
    min_col=1,
    min_row=2,
    max_row=4
)

chart2.add_data(
    data,
    titles_from_data=True
)

chart2.set_categories(cats)

chart2.height = 7
chart2.width = 12

dashboard.add_chart(
    chart2,
    "F20"
)

# ------------------------------------------------------------
# 26. CREATE CHART 3 - CUSTOMER TREND
# ------------------------------------------------------------

chart3 = LineChart()

chart3.title = "Customer Base Trend 2022-2024"
chart3.y_axis.title = "Customers"
chart3.x_axis.title = "Year"

data = Reference(
    wb["Yearly KPIs"],
    min_col=5,
    min_row=1,
    max_row=4
)

cats = Reference(
    wb["Yearly KPIs"],
    min_col=1,
    min_row=2,
    max_row=4
)

chart3.add_data(
    data,
    titles_from_data=True
)

chart3.set_categories(cats)

chart3.height = 7
chart3.width = 12

dashboard.add_chart(
    chart3,
    "A12"
)

# ------------------------------------------------------------
# 27. CREATE CHART 4 - STATION GROWTH
# ------------------------------------------------------------

chart4 = BarChart()

chart4.type = "col"
chart4.style = 10

chart4.title = "Station Network Growth"
chart4.y_axis.title = "Stations"
chart4.x_axis.title = "Year"

data = Reference(
    wb["Yearly KPIs"],
    min_col=2,
    min_row=1,
    max_row=4
)

cats = Reference(
    wb["Yearly KPIs"],
    min_col=1,
    min_row=2,
    max_row=4
)

chart4.add_data(
    data,
    titles_from_data=True
)

chart4.set_categories(cats)

chart4.height = 7
chart4.width = 12

dashboard.add_chart(
    chart4,
    "A27"
)

# ------------------------------------------------------------
# 28. CREATE CHART 5 - YOY REVENUE
# ------------------------------------------------------------

chart5 = BarChart()

chart5.type = "col"
chart5.title = "Revenue YoY Growth"
chart5.y_axis.title = "YoY Growth %"
chart5.x_axis.title = "Year"

data = Reference(
    wb["YoY Results"],
    min_col=9,
    min_row=1,
    max_row=4
)

cats = Reference(
    wb["YoY Results"],
    min_col=1,
    min_row=2,
    max_row=4
)

chart5.add_data(
    data,
    titles_from_data=True
)

chart5.set_categories(cats)

chart5.height = 7
chart5.width = 12

dashboard.add_chart(
    chart5,
    "F35"
)

# ------------------------------------------------------------
# 29. CREATE CHART 6 - YOY SESSIONS
# ------------------------------------------------------------

chart6 = BarChart()

chart6.type = "col"
chart6.title = "Sessions YoY Growth"
chart6.y_axis.title = "YoY Growth %"
chart6.x_axis.title = "Year"

data = Reference(
    wb["YoY Results"],
    min_col=8,
    min_row=1,
    max_row=4
)

cats = Reference(
    wb["YoY Results"],
    min_col=1,
    min_row=2,
    max_row=4
)

chart6.add_data(
    data,
    titles_from_data=True
)

chart6.set_categories(cats)

chart6.height = 7
chart6.width = 12

dashboard.add_chart(
    chart6,
    "A42"
)

# ------------------------------------------------------------
# 30. DASHBOARD COLUMN WIDTHS
# ------------------------------------------------------------

dashboard.column_dimensions["A"].width = 25
dashboard.column_dimensions["B"].width = 15
dashboard.column_dimensions["C"].width = 15
dashboard.column_dimensions["D"].width = 15
dashboard.column_dimensions["E"].width = 4
dashboard.column_dimensions["F"].width = 15
dashboard.column_dimensions["G"].width = 15
dashboard.column_dimensions["H"].width = 15

# ------------------------------------------------------------
# 31. SAVE FINAL WORKBOOK
# ------------------------------------------------------------

wb.save(OUTPUT_FILE)

# ------------------------------------------------------------
# 32. FINAL MESSAGE
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("TASK 12 PYTHON ANALYSIS COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nOutput file created:")
print(OUTPUT_FILE)

print("\nSheets created:")

for sheet in wb.sheetnames:
    print("-", sheet)

print("\nCharts created:")
print("- Revenue Trend")
print("- Sessions Trend")
print("- Customer Base Trend")
print("- Station Network Growth")
print("- Revenue YoY Growth")
print("- Sessions YoY Growth")

print("\nYour Python Task 12 deliverable is ready.")
print("=" * 60)

input("\nPress Enter to exit...")


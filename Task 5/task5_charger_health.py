import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# ============================================================
# TASK 5 - PREDICTIVE MAINTENANCE FOR EV CHARGERS
# ============================================================

print("\n" + "=" * 55)
print("TASK 5 - PREDICTIVE MAINTENANCE")
print("=" * 55)


# ------------------------------------------------------------
# 1. FIND AND LOAD EXCEL FILE
# ------------------------------------------------------------

print("\nLoading Excel file...")

# Use the actual workbook name
FILE_NAME = "Task 5 - Predictive Maintenance.xlsx"

try:
    excel_file = pd.ExcelFile(FILE_NAME)
except FileNotFoundError:
    print("\nERROR: Excel file not found.")
    print("Make sure the Excel file is in the same folder as this Python file.")
    raise

print("Available sheets:")
print(excel_file.sheet_names)


# ------------------------------------------------------------
# 2. READ RAW DATA
# ------------------------------------------------------------

if "Raw Data" in excel_file.sheet_names:
    df = pd.read_excel(
        FILE_NAME,
        sheet_name="Raw Data"
    )
else:
    df = pd.read_excel(
        FILE_NAME,
        sheet_name=excel_file.sheet_names[0]
    )

print("\nRaw data loaded successfully.")
print("Rows:", len(df))
print("Columns:", list(df.columns))


# ------------------------------------------------------------
# 3. CLEAN COLUMN NAMES
# ------------------------------------------------------------

df.columns = (
    df.columns
    .astype(str)
    .str.strip()
    .str.lower()
    .str.replace(" ", "_", regex=False)
)

print("\nCleaned columns:")
print(list(df.columns))


# ------------------------------------------------------------
# 4. FIND IMPORTANT COLUMNS
# ------------------------------------------------------------

def find_column(possible_names):

    for name in possible_names:
        if name in df.columns:
            return name

    return None


station_col = find_column([
    "station_id",
    "station"
])

charger_col = find_column([
    "charger_type",
    "charger"
])

duration_col = find_column([
    "session_duration_minutes",
    "duration_min",
    "duration",
    "session_duration"
])

energy_col = find_column([
    "energy_kwh",
    "energy"
])

cost_col = find_column([
    "total_cost",
    "cost"
])


print("\nDetected columns:")
print("Station:", station_col)
print("Charger:", charger_col)
print("Duration:", duration_col)
print("Energy:", energy_col)
print("Cost:", cost_col)


# ------------------------------------------------------------
# 5. CHECK REQUIRED COLUMNS
# ------------------------------------------------------------

required_columns = [
    station_col,
    charger_col,
    duration_col,
    energy_col,
    cost_col
]

if any(col is None for col in required_columns):

    print("\nERROR: One or more required columns were not found.")
    print("Please check your Raw Data column names.")
    raise ValueError("Required columns missing.")


# ------------------------------------------------------------
# 6. CONVERT NUMERIC COLUMNS
# ------------------------------------------------------------

for col in [
    duration_col,
    energy_col,
    cost_col
]:

    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )


# ------------------------------------------------------------
# 7. REMOVE MISSING VALUES
# ------------------------------------------------------------

important_cols = [
    station_col,
    charger_col,
    duration_col,
    energy_col,
    cost_col
]

df = df.dropna(
    subset=important_cols
).copy()

print("\nClean data rows:", len(df))


# ------------------------------------------------------------
# 8. CREATE CHARGER HEALTH TABLE
# ------------------------------------------------------------

print("\nCreating charger health table...")


charger_health = (
    df.groupby(
        [station_col, charger_col]
    )
    .agg(
        total_sessions=(duration_col, "count"),
        average_duration=(duration_col, "mean"),
        total_energy=(energy_col, "sum"),
        average_energy=(energy_col, "mean"),
        average_cost=(cost_col, "mean")
    )
    .reset_index()
)


print("\nCharger health table created.")

print(
    charger_health.head()
)


# ------------------------------------------------------------
# 9. NORMALIZATION FUNCTION
# ------------------------------------------------------------

def normalize(series):

    minimum = series.min()
    maximum = series.max()

    if maximum == minimum:

        return pd.Series(
            50,
            index=series.index
        )

    return (
        (series - minimum)
        /
        (maximum - minimum)
        * 100
    )


# ------------------------------------------------------------
# 10. CREATE HEALTH SCORE
# ------------------------------------------------------------

session_score = normalize(
    charger_health["total_sessions"]
)

energy_score = normalize(
    charger_health["total_energy"]
)

duration_score = normalize(
    charger_health["average_duration"]
)


# Health score
# Higher usage and energy contribute positively.
# Higher average duration reduces the score.

charger_health["health_score"] = (

    40

    + session_score * 0.20

    + energy_score * 0.20

    - duration_score * 0.20

)


charger_health["health_score"] = (
    charger_health["health_score"]
    .clip(0, 100)
    .round(0)
)


# ------------------------------------------------------------
# 11. HEALTH STATUS
# ------------------------------------------------------------

def health_status(score):

    if score >= 70:
        return "Healthy"

    elif score >= 50:
        return "Monitor"

    else:
        return "Maintenance Required"


charger_health["health_status"] = (
    charger_health["health_score"]
    .apply(health_status)
)


# ------------------------------------------------------------
# 12. RISK LEVEL
# ------------------------------------------------------------

def risk_level(score):

    if score < 50:
        return "High"

    elif score < 70:
        return "Medium"

    else:
        return "Low"


charger_health["risk_level"] = (
    charger_health["health_score"]
    .apply(risk_level)
)


# ------------------------------------------------------------
# 13. PREDICTIVE MAINTENANCE MODEL
# ------------------------------------------------------------

print("\nBuilding predictive maintenance model...")


features = [
    "total_sessions",
    "average_duration",
    "total_energy",
    "average_energy",
    "average_cost"
]


X = charger_health[features].copy()


# Target:
# 1 = Maintenance Required
# 0 = Normal

y = (
    charger_health["health_status"]
    == "Maintenance Required"
).astype(int)


print("\nTarget distribution:")
print(y.value_counts())


# ------------------------------------------------------------
# 14. HANDLE TARGET CLASSES
# ------------------------------------------------------------

if y.nunique() < 2:

    print(
        "\nOnly one maintenance class detected."
    )

    print(
        "Creating maintenance target using health score..."
    )

    y = (
        charger_health["health_score"] < 60
    ).astype(int)


# ------------------------------------------------------------
# 15. TRAIN / TEST SPLIT
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.25,

    random_state=42,

    stratify=y if y.nunique() > 1 else None

)


# ------------------------------------------------------------
# 16. RANDOM FOREST MODEL
# ------------------------------------------------------------

model = RandomForestClassifier(

    n_estimators=200,

    random_state=42,

    class_weight="balanced"

)


model.fit(
    X_train,
    y_train
)


# ------------------------------------------------------------
# 17. MODEL PREDICTIONS
# ------------------------------------------------------------

y_pred = model.predict(
    X_test
)


# ------------------------------------------------------------
# 18. MODEL ACCURACY
# ------------------------------------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)


print("\n" + "=" * 55)
print("MODEL RESULTS")
print("=" * 55)

print(
    "\nModel Accuracy:",
    round(accuracy * 100, 2),
    "%"
)


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# ------------------------------------------------------------
# 19. PREDICT ALL CHARGER GROUPS
# ------------------------------------------------------------

charger_health["maintenance_prediction"] = (
    model.predict(X)
)


charger_health["maintenance_prediction"] = (
    charger_health[
        "maintenance_prediction"
    ]
    .map({
        0: "Normal",
        1: "Maintenance Required"
    })
)


# ------------------------------------------------------------
# 20. MAINTENANCE PRIORITY
# ------------------------------------------------------------

def priority(score):

    if score < 50:
        return "High"

    elif score < 70:
        return "Medium"

    else:
        return "Low"


charger_health["maintenance_priority"] = (
    charger_health["health_score"]
    .apply(priority)
)


# ------------------------------------------------------------
# 21. FEATURE IMPORTANCE
# ------------------------------------------------------------

importance = pd.DataFrame({

    "Feature": features,

    "Importance": model.feature_importances_

})


importance = (
    importance
    .sort_values(
        "Importance",
        ascending=False
    )
    .reset_index(drop=True)
)


print("\nFeature Importance:")

print(
    importance
)


# ------------------------------------------------------------
# 22. MAINTENANCE PRIORITY LIST
# ------------------------------------------------------------

priority_order = {

    "High": 1,

    "Medium": 2,

    "Low": 3

}


maintenance_priority = (
    charger_health.copy()
)


maintenance_priority["priority_order"] = (
    maintenance_priority[
        "maintenance_priority"
    ]
    .map(priority_order)
)


maintenance_priority = (
    maintenance_priority
    .sort_values(
        [
            "priority_order",
            "health_score"
        ]
    )
    .drop(
        columns=["priority_order"]
    )
)


# ------------------------------------------------------------
# 23. MODEL SUMMARY
# ------------------------------------------------------------

summary = pd.DataFrame({

    "Metric": [

        "Model Accuracy",

        "Total Charger Groups",

        "High Priority",

        "Medium Priority",

        "Low Priority"

    ],

    "Value": [

        round(
            accuracy * 100,
            2
        ),

        len(charger_health),

        (
            charger_health[
                "maintenance_priority"
            ] == "High"
        ).sum(),

        (
            charger_health[
                "maintenance_priority"
            ] == "Medium"
        ).sum(),

        (
            charger_health[
                "maintenance_priority"
            ] == "Low"
        ).sum()

    ]

})


# ------------------------------------------------------------
# 24. SAVE EXCEL RESULTS
# ------------------------------------------------------------

OUTPUT_FILE = "task5_python_results.xlsx"

print("\nSaving Python results...")


with pd.ExcelWriter(
    OUTPUT_FILE,
    engine="openpyxl"
) as writer:

    charger_health.to_excel(

        writer,

        sheet_name="Charger Health",

        index=False

    )


    maintenance_priority.to_excel(

        writer,

        sheet_name="Maintenance Priority",

        index=False

    )


    importance.to_excel(

        writer,

        sheet_name="Feature Importance",

        index=False

    )


    summary.to_excel(

        writer,

        sheet_name="Model Summary",

        index=False

    )


# ------------------------------------------------------------
# 25. FEATURE IMPORTANCE CHART
# ------------------------------------------------------------

plt.figure(
    figsize=(9, 5)
)


plt.barh(

    importance["Feature"],

    importance["Importance"]

)


plt.xlabel(
    "Importance"
)

plt.ylabel(
    "Feature"
)

plt.title(
    "Predictive Maintenance - Feature Importance"
)


plt.gca().invert_yaxis()


plt.tight_layout()


plt.savefig(

    "task5_feature_importance.png",

    dpi=300,

    bbox_inches="tight"

)


plt.show()


# ------------------------------------------------------------
# 26. MAINTENANCE PRIORITY CHART
# ------------------------------------------------------------

priority_counts = (

    charger_health[
        "maintenance_priority"
    ]

    .value_counts()

    .reindex(
        [
            "High",
            "Medium",
            "Low"
        ],
        fill_value=0
    )

)


plt.figure(
    figsize=(8, 5)
)


plt.bar(

    priority_counts.index,

    priority_counts.values

)


plt.xlabel(
    "Maintenance Priority"
)

plt.ylabel(
    "Number of Charger Groups"
)

plt.title(
    "Charger Maintenance Priority"
)


plt.tight_layout()


plt.savefig(

    "task5_maintenance_priority.png",

    dpi=300,

    bbox_inches="tight"

)


plt.show()


# ------------------------------------------------------------
# 27. FINAL MESSAGE
# ------------------------------------------------------------

print("\n" + "=" * 55)
print("TASK 5 PYTHON ANALYSIS COMPLETED SUCCESSFULLY")
print("=" * 55)

print("\nFiles created:")

print("1. task5_python_results.xlsx")

print("2. task5_feature_importance.png")

print("3. task5_maintenance_priority.png")

print("\nYou can now use these files for Task 5 submission.")
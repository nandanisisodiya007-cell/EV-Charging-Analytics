import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Load the EV charging dataset
df = pd.read_csv("EV_Forecasting_Project/EV_Charging_demand_Forecasting.csv")

print("Dataset loaded successfully")

# Rename columns
df.columns=["Date","Energy_kWh"]

# Convert Date column to proper date format 
df["Date"] = pd.to_datetime(df["Date"], dayfirst =True)

# Convert energy column to numeric 
df["Energy_kWh"] = pd.to_numeric(df["Energy_kWh"], errors="coerce")

# Check the cleaned data
print ("\nCleaned dataset:")
print(df.head())

print ("\nColumn names:")
print(df.columns)

print ("\nDataset shape:")
print(df.shape)

print ("\nMissing values:")
print(df.isnull().sum())

# Plot EV charging demand over time
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))

plt.plot(df["Date"], df["Energy_kWh"])

plt.title("EV Charging Demand Over Time")
plt.xlabel("Date")
plt.ylabel("Energy (kWh)")

# Show only selected dates on x-axis
plt.xticks (df["Date"][::100], rotation=45)

plt.tight_layout()
plt.show()

# ==========================================
# EV CHARGING DEMAND FORECASTING
# ==========================================


import numpy as np

# Create time-based features
df["Day"] = df["Date"].dt.day
df["Month"] = df["Date"].dt.month
df["Year"] = df["Date"].dt.year
df["DayOfWeek"] = df["Date"].dt.dayofweek
df["DayOfYear"] = df["Date"].dt.dayofyear

# Create lag features
df["Lag_1"] = df["Energy_kWh"].shift(1)
df["Lag_7"] = df["Energy_kWh"].shift(7)

# Remove rows created by lagging
df = df.dropna()

# Features used for forecasting
features = [
    "Day",
    "Month",
    "Year",
    "DayOfWeek",
    "DayOfYear",
    "Lag_1",
    "Lag_7"
]

X = df[features]
y = df["Energy_kWh"]

# Split data chronologically
train_size = int(len(df) * 0.8)

X_train = X.iloc[:train_size]
X_test = X.iloc[train_size:]

y_train = y.iloc[:train_size]
y_test = y.iloc[train_size:]

print("\nTraining data:", len(X_train))
print("Testing data:", len(X_test))
# ==========================================
# SARIMA FORECASTING
# ==========================================

# Use the Energy_kWh time series
sarima_data = df.set_index("Date")["Energy_kWh"]

# Split data chronologically
sarima_train_size = int(len(sarima_data) * 0.8)

sarima_train = sarima_data.iloc[:sarima_train_size]
sarima_test = sarima_data.iloc[sarima_train_size:]

print("\nSARIMA Training data:", len(sarima_train))
print("SARIMA Testing data:", len(sarima_test))

# Create SARIMA model
sarima_model = SARIMAX(
    sarima_train,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 7),
    enforce_stationarity=False,
    enforce_invertibility=False
)

# Train SARIMA model
sarima_result = sarima_model.fit(disp=False)

# Forecast the testing period
sarima_forecast = sarima_result.forecast(steps=len(sarima_test))

print("\nSARIMA model completed successfully")

# ==========================================
# SARIMA ACCURACY METRICS
# ==========================================

# Calculate RMSE
sarima_rmse = np.sqrt(np.mean((sarima_test - sarima_forecast) ** 2))

# Calculate MAPE
sarima_mape = np.mean(
    np.abs((sarima_test - sarima_forecast) / sarima_test)
) * 100

print("\nSARIMA Accuracy")
print("-------------------")
print("RMSE:", round(sarima_rmse, 2))
print("MAPE:", round(sarima_mape, 2), "%")
# ==========================================
# SARIMA ACTUAL VS FORECAST
# ==========================================

plt.figure(figsize=(12, 6))

plt.plot(
    sarima_test.index,
    sarima_test,
    label="Actual Demand"
)

plt.plot(
    sarima_test.index,
    sarima_forecast,
    label="SARIMA Forecast"
)

plt.title("SARIMA: Actual vs Forecast EV Charging Demand")
plt.xlabel("Date")
plt.ylabel("Energy (kWh)")
plt.legend()

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
# ==========================================
# PROPHET EV CHARGING DEMAND FORECASTING
# ==========================================

from prophet import Prophet
import numpy as np
import matplotlib.pyplot as plt

# Prepare data for Prophet
# Prophet requires columns named "ds" and "y"

prophet_df = df[["Date", "Energy_kWh"]].copy()

prophet_df = prophet_df.dropna()

prophet_df = prophet_df.rename(columns={
    "Date": "ds",
    "Energy_kWh": "y"
})

# Make sure dates are sorted
prophet_df = prophet_df.sort_values("ds")

# Split data chronologically
prophet_train_size = int(len(prophet_df) * 0.8)

prophet_train = prophet_df.iloc[:prophet_train_size]
prophet_test = prophet_df.iloc[prophet_train_size:]

print("\n==========================================")
print("PROPHET FORECASTING")
print("==========================================")

print("Prophet Training data:", len(prophet_train))
print("Prophet Testing data:", len(prophet_test))

# Create Prophet model
prophet_model = Prophet(
    daily_seasonality=True,
    weekly_seasonality=True,
    yearly_seasonality=True
)

# Train Prophet model
prophet_model.fit(prophet_train)

print("Prophet model trained successfully")

# Create dataframe containing test dates
future = prophet_test[["ds"]].copy()

# Generate forecast
prophet_forecast = prophet_model.predict(future)

# Actual and predicted values
y_actual_prophet = prophet_test["y"].values
y_pred_prophet = prophet_forecast["yhat"].values

# ==========================================
# ACCURACY METRICS
# ==========================================

# RMSE
prophet_rmse = np.sqrt(
    np.mean((y_actual_prophet - y_pred_prophet) ** 2)
)

# MAPE
prophet_mape = np.mean(
    np.abs(
        (y_actual_prophet - y_pred_prophet)
        / y_actual_prophet
    )
) * 100

print("\nProphet Forecasting Results")
print("---------------------------")
print("RMSE:", round(prophet_rmse, 2))
print("MAPE:", round(prophet_mape, 2), "%")

# ==========================================
# PLOT ACTUAL VS PROPHET FORECAST
# ==========================================

plt.figure(figsize=(12, 6))

plt.plot(
    prophet_test["ds"],
    y_actual_prophet,
    label="Actual Demand"
)

plt.plot(
    prophet_test["ds"],
    y_pred_prophet,
    label="Prophet Forecast"
)

plt.title("Prophet: Actual vs Forecast EV Charging Demand")
plt.xlabel("Date")
plt.ylabel("Energy (kWh)")
plt.legend()

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
# ==========================================
# XGBOOST FORECASTING
# WITHOUT SCIKIT-LEARN
# ==========================================
import xgboost as xgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("\nXGBOOST FORECASTING")
print("===================")

# Load the original dataset again
xgb_df = pd.read_csv(
    "EV_Forecasting_Project/EV_Charging_demand_Forecasting.csv"
)

# Rename columns
xgb_df.columns = ["Date", "Energy_kWh"]

# Convert columns
xgb_df["Date"] = pd.to_datetime(
    xgb_df["Date"],
    dayfirst=True
)

xgb_df["Energy_kWh"] = pd.to_numeric(
    xgb_df["Energy_kWh"],
    errors="coerce"
)

# Create time features
xgb_df["Day"] = xgb_df["Date"].dt.day
xgb_df["Month"] = xgb_df["Date"].dt.month
xgb_df["Year"] = xgb_df["Date"].dt.year
xgb_df["DayOfWeek"] = xgb_df["Date"].dt.dayofweek
xgb_df["DayOfYear"] = xgb_df["Date"].dt.dayofyear

# Create lag features
xgb_df["Lag_1"] = xgb_df["Energy_kWh"].shift(1)
xgb_df["Lag_7"] = xgb_df["Energy_kWh"].shift(7)

# Remove missing values
xgb_df = xgb_df.dropna()

# Features
features = [
    "Day",
    "Month",
    "Year",
    "DayOfWeek",
    "DayOfYear",
    "Lag_1",
    "Lag_7"
]

X = xgb_df[features]
y = xgb_df["Energy_kWh"]

# Chronological 80/20 split
train_size = int(len(xgb_df) * 0.8)

X_train = X.iloc[:train_size]
X_test = X.iloc[train_size:]

y_train = y.iloc[:train_size]
y_test = y.iloc[train_size:]

print("XGBoost Training data:", len(X_train))
print("XGBoost Testing data:", len(X_test))

# Convert data to XGBoost DMatrix
train_data = xgb.DMatrix(
    X_train,
    label=y_train
)

#XGBoost parameters 
params ={
    "objective": "reg:squarederror",
    "max_depth": 6,
    "learning_rate": 0.1,
    "seed": 42
}

# Train XGBoost model
model_xgb = xgb.train(
    params,
    train_data,
    num_boost_round=100
)

print("\nXGBoost model trained successfully")

# Convert training data into XGBoost format
train_data = xgb.DMatrix(
    X_train,
    label=y_train
)

# Convert test data into XGBoost format 
test_data =xgb.DMatrix(
    X_test,
    label=y_test
)

# Make predictions
y_pred_xgb = model_xgb.predict(test_data)

# ==========================================
# CALCULATE RMSE WITHOUT SKLEARN
# ==========================================

rmse_xgb = np.sqrt(
    np.mean((y_test.values - y_pred_xgb) ** 2)
)

# ==========================================
# CALCULATE MAPE WITHOUT SKLEARN
# ==========================================

mape_xgb = np.mean(
    np.abs(
        (y_test.values - y_pred_xgb)
        / y_test.values
    )
) * 100

print("\nXGBoost Forecasting Results")
print("---------------------------")
print("RMSE:", round(rmse_xgb, 2))
print("MAPE:", round(mape_xgb, 2), "%")

# ==========================================
# PLOT ACTUAL VS PREDICTED
# ==========================================

plt.figure(figsize=(12, 6))

plt.plot(
    xgb_df["Date"].iloc[train_size:],
    y_test,
    label="Actual Demand"
)

plt.plot(
    xgb_df["Date"].iloc[train_size:],
    y_pred_xgb,
    label="XGBoost Forecast"
)

plt.title(
    "XGBoost: Actual vs Forecast EV Charging Demand"
)

plt.xlabel("Date")
plt.ylabel("Energy (kWh)")

plt.legend()
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# ============================================================
# FUTURE FORECAST DASHBOARD - 30 / 90 / 365 DAYS
# ============================================================

print("\n" + "=" * 60)
print("FUTURE EV CHARGING DEMAND FORECAST")
print("=" * 60)

# ------------------------------------------------------------
# FIND DATE COLUMN
# ------------------------------------------------------------

date_column = None

for col in df.columns:
    col_name = str(col).strip().lower()

    if col_name in ["date", "datetime", "timestamp", "ds"]:
        date_column = col
        break

if date_column is None:
    print("Available columns:")
    print(df.columns.tolist())
    raise ValueError("Date column not found.")

# ------------------------------------------------------------
# FIND ENERGY COLUMN
# ------------------------------------------------------------

energy_column = None

for col in df.columns:
    col_name = str(col).strip().lower()

    if "energy" in col_name:
        energy_column = col
        break

if energy_column is None:
    print("Available columns:")
    print(df.columns.tolist())
    raise ValueError("Energy column not found.")

print("Date column:", date_column)
print("Energy column:", energy_column)

# ------------------------------------------------------------
# PREPARE PROPHET DATA
# ------------------------------------------------------------

prophet_full = pd.DataFrame({
    "ds": pd.to_datetime(df[date_column], errors="coerce"),
    "y": pd.to_numeric(df[energy_column], errors="coerce")
})

prophet_full = prophet_full.dropna()
prophet_full = prophet_full.sort_values("ds")

print("Historical data:", len(prophet_full))
print("Last historical date:", prophet_full["ds"].max())

# ------------------------------------------------------------
# TRAIN PROPHET MODEL
# ------------------------------------------------------------

dashboard_model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False
)

dashboard_model.fit(prophet_full)

print("\nDashboard Prophet model trained successfully")

# ------------------------------------------------------------
# FORECAST NEXT 365 DAYS
# ------------------------------------------------------------

future = dashboard_model.make_future_dataframe(
    periods=365,
    freq="D"
)

forecast = dashboard_model.predict(future)

# Only future dates
last_date = prophet_full["ds"].max()

future_forecast = forecast[
    forecast["ds"] > last_date
].copy()

# ------------------------------------------------------------
# 30 / 90 / 365 DAY FORECASTS
# ------------------------------------------------------------

forecast_30 = future_forecast.head(30)
forecast_90 = future_forecast.head(90)
forecast_365 = future_forecast.head(365)

# ------------------------------------------------------------
# DISPLAY RESULTS
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("30 DAY FORECAST")
print("=" * 60)

print(
    "Average demand:",
    round(forecast_30["yhat"].mean(), 2),
    "kWh"
)

print(
    "Total demand:",
    round(forecast_30["yhat"].sum(), 2),
    "kWh"
)

print("\n" + "=" * 60)
print("90 DAY FORECAST")
print("=" * 60)

print(
    "Average demand:",
    round(forecast_90["yhat"].mean(), 2),
    "kWh"
)

print(
    "Total demand:",
    round(forecast_90["yhat"].sum(), 2),
    "kWh"
)

print("\n" + "=" * 60)
print("365 DAY FORECAST")
print("=" * 60)

print(
    "Average demand:",
    round(forecast_365["yhat"].mean(), 2),
    "kWh"
)

print(
    "Total demand:",
    round(forecast_365["yhat"].sum(), 2),
    "kWh"
)

# ------------------------------------------------------------
# FORECAST DASHBOARD GRAPH
# ------------------------------------------------------------

plt.figure(figsize=(15, 8))

plt.plot(
    prophet_full["ds"],
    prophet_full["y"],
    label="Historical Demand"
)

plt.plot(
    forecast_30["ds"],
    forecast_30["yhat"],
    label="30-Day Forecast"
)

plt.plot(
    forecast_90["ds"],
    forecast_90["yhat"],
    label="90-Day Forecast"
)

plt.plot(
    forecast_365["ds"],
    forecast_365["yhat"],
    label="365-Day Forecast"
)

plt.axvline(
    last_date,
    linestyle="--",
    label="Forecast Start"
)

plt.title(
    "EV Charging Demand Forecast Dashboard"
)

plt.xlabel("Date")

plt.ylabel(
    "Energy Demand (kWh)"
)

plt.legend()

plt.grid(True, alpha=0.3)

plt.xticks(rotation=45)

plt.tight_layout()

plt.show()

print("\nForecast dashboard completed successfully.")

# ============================================================
# SEASONAL AND WEEKLY PATTERN ANALYSIS
# ============================================================

print("\n" + "=" * 65)
print("SEASONAL AND WEEKLY PATTERN ANALYSIS")
print("=" * 65)

# ------------------------------------------------------------
# USE THE ORIGINAL DATE AND ENERGY COLUMNS
# ------------------------------------------------------------

pattern_df = df.copy()

# Display columns so we can verify them
print("\nAvailable columns:")
print(pattern_df.columns.tolist())

# Find Date column
date_column = None

for col in pattern_df.columns:
    if str(col).strip().lower() == "date":
        date_column = col
        break

# Find Energy column automatically
energy_column = None

for col in pattern_df.columns:
    col_name = str(col).strip().lower().replace(" ", "_")

    if "energy" in col_name:
        energy_column = col
        break

if date_column is None:
    raise ValueError("Date column not found.")

if energy_column is None:
    raise ValueError(
        "Energy column not found. Check the column names printed above."
    )

print("\nUsing Date column:", date_column)
print("Using Energy column:", energy_column)


# ------------------------------------------------------------
# CLEAN DATA
# ------------------------------------------------------------

pattern_df[date_column] = pd.to_datetime(
    pattern_df[date_column],
    errors="coerce"
)

pattern_df[energy_column] = pd.to_numeric(
    pattern_df[energy_column],
    errors="coerce"
)

pattern_df = pattern_df.dropna(
    subset=[date_column, energy_column]
)


# ------------------------------------------------------------
# CREATE TIME FEATURES
# ------------------------------------------------------------

pattern_df["Month"] = pattern_df[date_column].dt.month

pattern_df["Month_Name"] = (
    pattern_df[date_column].dt.month_name()
)

pattern_df["DayOfWeek"] = (
    pattern_df[date_column].dt.dayofweek
)

pattern_df["Day_Name"] = (
    pattern_df[date_column].dt.day_name()
)


# ============================================================
# MONTHLY / SEASONAL ANALYSIS
# ============================================================

monthly_demand = (
    pattern_df
    .groupby(
        ["Month", "Month_Name"]
    )[energy_column]
    .mean()
    .reset_index()
)

monthly_demand = monthly_demand.sort_values(
    "Month"
)

print("\nAverage Energy Demand by Month:")

print(
    monthly_demand[
        ["Month_Name", energy_column]
    ].to_string(index=False)
)


# Highest and lowest month

highest_month = monthly_demand.loc[
    monthly_demand[energy_column].idxmax()
]

lowest_month = monthly_demand.loc[
    monthly_demand[energy_column].idxmin()
]

print("\nHighest average demand month:")
print(
    highest_month["Month_Name"],
    "-",
    round(highest_month[energy_column], 2),
    "kWh"
)

print("\nLowest average demand month:")
print(
    lowest_month["Month_Name"],
    "-",
    round(lowest_month[energy_column], 2),
    "kWh"
)


# ------------------------------------------------------------
# MONTHLY GRAPH
# ------------------------------------------------------------

plt.figure(figsize=(12, 6))

plt.plot(
    monthly_demand["Month_Name"],
    monthly_demand[energy_column],
    marker="o"
)

plt.title(
    "Average EV Charging Demand by Month"
)

plt.xlabel("Month")

plt.ylabel(
    "Average Energy Demand (kWh)"
)

plt.xticks(rotation=45)

plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.show()


# ============================================================
# WEEKLY / DAY-OF-WEEK ANALYSIS
# ============================================================

weekly_demand = (
    pattern_df
    .groupby(
        ["DayOfWeek", "Day_Name"]
    )[energy_column]
    .mean()
    .reset_index()
)

weekly_demand = weekly_demand.sort_values(
    "DayOfWeek"
)

print("\nAverage Energy Demand by Day of Week:")

print(
    weekly_demand[
        ["Day_Name", energy_column]
    ].to_string(index=False)
)


# Highest and lowest day

highest_day = weekly_demand.loc[
    weekly_demand[energy_column].idxmax()
]

lowest_day = weekly_demand.loc[
    weekly_demand[energy_column].idxmin()
]

print("\nHighest average demand day:")
print(
    highest_day["Day_Name"],
    "-",
    round(highest_day[energy_column], 2),
    "kWh"
)

print("\nLowest average demand day:")
print(
    lowest_day["Day_Name"],
    "-",
    round(lowest_day[energy_column], 2),
    "kWh"
)


# ------------------------------------------------------------
# WEEKLY GRAPH
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

plt.bar(
    weekly_demand["Day_Name"],
    weekly_demand[energy_column]
)

plt.title(
    "Average EV Charging Demand by Day of Week"
)

plt.xlabel("Day of Week")

plt.ylabel(
    "Average Energy Demand (kWh)"
)

plt.xticks(rotation=45)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.show()


# ============================================================
# SAVE RESULTS
# ============================================================

monthly_demand.to_csv(
    "monthly_EV_charging_demand.csv",
    index=False
)

weekly_demand.to_csv(
    "weekly_EV_charging_demand.csv",
    index=False
)

print("\n" + "=" * 65)
print("PATTERN ANALYSIS COMPLETED SUCCESSFULLY")
print("=" * 65)

print(
    "Highest-demand month:",
    highest_month["Month_Name"]
)

print(
    "Lowest-demand month:",
    lowest_month["Month_Name"]
)

print(
    "Highest-demand day:",
    highest_day["Day_Name"]
)

print(
    "Lowest-demand day:",
    lowest_day["Day_Name"]
)
# ============================================================
# FINAL TASK 1 DASHBOARD / REPORT
# ============================================================

print("\n" + "=" * 70)
print("FINAL TASK 1 - EV CHARGING DEMAND FORECASTING DASHBOARD")
print("=" * 70)

# ------------------------------------------------------------
# 1. MODEL PERFORMANCE SUMMARY
# ------------------------------------------------------------

model_results = pd.DataFrame({
    "Model": ["SARIMA", "Prophet", "XGBoost"],
    "RMSE": [1314.31, 3205.26, 1060.93],
    "MAPE (%)": [5.18, 13.76, 3.75]
})

print("\nMODEL PERFORMANCE")
print("-" * 50)
print(model_results.to_string(index=False))

# ------------------------------------------------------------
# 2. FUTURE FORECAST SUMMARY
# ------------------------------------------------------------

forecast_summary = pd.DataFrame({
    "Forecast Period": ["30 Days", "90 Days", "365 Days"],
    "Average Demand (kWh)": [28336.89, 28494.91, 28644.33],
    "Total Demand (kWh)": [850106.85, 2564541.63, 10455179.27]
})

print("\nFUTURE FORECAST SUMMARY")
print("-" * 70)
print(forecast_summary.to_string(index=False))

# ------------------------------------------------------------
# 3. CREATE FINAL DASHBOARD - MODEL PERFORMANCE
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

plt.bar(
    model_results["Model"],
    model_results["MAPE (%)"]
)

plt.title("Forecasting Model Comparison - MAPE")
plt.xlabel("Forecasting Model")
plt.ylabel("MAPE (%)")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig(
    "model_comparison_mape.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ------------------------------------------------------------
# 4. RMSE COMPARISON
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

plt.bar(
    model_results["Model"],
    model_results["RMSE"]
)

plt.title("Forecasting Model Comparison - RMSE")
plt.xlabel("Forecasting Model")
plt.ylabel("RMSE")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig(
    "model_comparison_rmse.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ------------------------------------------------------------
# 5. FUTURE DEMAND FORECAST SUMMARY
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

plt.bar(
    forecast_summary["Forecast Period"],
    forecast_summary["Average Demand (kWh)"]
)

plt.title("Average EV Charging Demand Forecast")
plt.xlabel("Forecast Period")
plt.ylabel("Average Demand (kWh)")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig(
    "future_demand_forecast.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ------------------------------------------------------------
# 6. FORECAST TOTAL DEMAND
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

plt.bar(
    forecast_summary["Forecast Period"],
    forecast_summary["Total Demand (kWh)"]
)

plt.title("Total Forecasted EV Charging Demand")
plt.xlabel("Forecast Period")
plt.ylabel("Total Demand (kWh)")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig(
    "total_forecast_demand.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ------------------------------------------------------------
# 7. SAVE FINAL REPORT TABLES
# ------------------------------------------------------------

model_results.to_csv(
    "model_performance_summary.csv",
    index=False
)

forecast_summary.to_csv(
    "future_forecast_summary.csv",
    index=False
)

# ------------------------------------------------------------
# 8. FINAL PROJECT SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FINAL TASK 1 SUMMARY")
print("=" * 70)

print("\nHistorical Pattern Analysis:")
print("Highest-demand month :", highest_month["Month_Name"])
print("Lowest-demand month :", lowest_month["Month_Name"])
print("Highest-demand day :", highest_day["Day_Name"])
print("Lowest-demand day :", lowest_day["Day_Name"])

print("\nModel Performance:")
print("SARIMA -> RMSE: 1314.31 | MAPE: 5.18%")
print("Prophet -> RMSE: 3205.26 | MAPE: 13.76%")
print("XGBoost -> RMSE: 1060.93 | MAPE: 3.75%")

print("\nFuture Forecast:")
print("30 Days -> Average:", 28336.89, "kWh | Total:", 850106.85, "kWh")
print("90 Days -> Average:", 28494.91, "kWh | Total:", 2564541.63, "kWh")
print("365 Days -> Average:", 28644.33, "kWh | Total:", 10455179.27, "kWh")

print("\nDashboard charts saved successfully.")
print("Final Task 1 report completed successfully.")

print("=" * 70)
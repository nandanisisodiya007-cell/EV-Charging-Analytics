Task 16: Energy Consumption & Grid Load Analysis

Project Overview

This project analyzes electric vehicle (EV) charging session data to understand energy consumption across stations, regions, and hours of the day. It supports energy planning, demand monitoring, and sustainability analysis.

Objectives

- Calculate total energy delivered per station and region.
- Analyze energy consumption by hour of the day.
- Summarize daily energy consumption.
- Produce a simple rolling-average reference for daily energy.
- Identify high-energy stations and peak energy-consumption hours.

Tools Used

- Python
- Pandas
- Matplotlib
- Excel dataset

Dataset

The analysis uses the "Sessions" worksheet from "ev_charging_dataset.xlsx".

The source dataset is not included in this repository. Place the dataset in the project folder before running the script.

Methodology

1. Load charging session data.
2. Convert timestamps and energy values into appropriate formats.
3. Remove rows missing essential analysis fields and exclude negative energy values.
4. Aggregate energy and session counts by station, region, hour, and day.
5. Calculate a seven-day rolling-average reference for daily energy.
6. Export summary CSV files and visualizations into the "results" folder.

Key Findings

- Total charging sessions analyzed: 294,024.
- Total energy delivered: 13,808,745.8 kWh.
- Stations analyzed: 33.
- Regions analyzed: 9.
- The highest aggregated hourly energy was observed at hour 10 (10:00–10:59).
- Station "STN_002" recorded the highest total energy among the stations analyzed.

Output Files

The "results" folder contains:

- "overall_energy_summary.csv"
- "station_energy_summary.csv"
- "regional_energy_summary.csv"
- "hourly_load_summary.csv"
- "daily_energy_summary.csv"
- "daily_energy_forecast.csv"
- "regional_energy_chart.png"
- "hourly_load_chart.png"
- "daily_energy_forecast_chart.png"

How to Run

Install the required libraries:

pip install -r requirements.txt

Run the analysis:

python energy_analysis.py

Limitations

The hourly analysis aggregates energy by session start hour. It does not calculate simultaneous grid power demand or actual grid stress. The seven-day rolling average is a simple reference, not a validated forecasting model.

Conclusion

This project provides an overview of EV charging energy patterns and regional differences. The findings can help guide further investigation into station capacity, charging schedules, and energy infrastructure planning.

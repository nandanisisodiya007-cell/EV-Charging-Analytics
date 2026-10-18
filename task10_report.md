Task 10: Session Duration & Energy Consumption Prediction

1. Objective

The objective of this project is to develop machine-learning models to predict EV charging session duration and energy consumption.

2. Dataset

The project uses EV charging data containing session records, station information, and charger information. The workbook includes the "sessions", "stations", and "chargers" sheets.

3. Methodology

- Loaded the Excel data using Pandas.
- Prepared the input features and prediction targets.
- Split the data into training and testing sets.
- Trained Random Forest Regression models.
- Evaluated the models using MAE, RMSE, and R².
- Generated prediction comparison charts and feature-importance charts.

4. Models Developed

Model 1: Session Duration Prediction

Target: "session_duration_minutes"

Model 2: Energy Consumption Prediction

Target: "energy_kwh"

5. Model Evaluation

The initial test results were:

Target| MAE| RMSE| R²
Session Duration| 1.907| 2.615| 0.964
Energy Consumption| 3.762| 4.169| 0.937

These initial results indicate that the models explain a substantial proportion of variation in the test data. Further validation is required before using them for operational decisions.

6. Visualisations

The project generates:

- Actual-versus-predicted comparison charts.
- Model R² comparison chart.
- Feature-importance charts for both prediction targets.

7. Business Applications

- Estimate charging session durations.
- Support charging-station capacity planning.
- Improve energy-demand estimation.
- Help operators understand factors associated with charging duration and energy consumption.

8. Limitations

Predictions depend on the quality and representativeness of the data. The models require additional checks for data leakage, generalisation to new stations, and performance on future sessions.

9. Conclusion

Two Random Forest regression models were developed to predict session duration and energy consumption. Evaluation metrics and visualisations were generated to assess their initial performance. Further validation is recommended before deployment.

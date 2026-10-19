Task 11: EV Charging Anomaly Detection (Fraud / Misuse)

Project Overview

This project uses machine learning to identify unusual patterns in electric vehicle (EV) charging sessions. It uses the Isolation Forest algorithm to flag potentially anomalous sessions for further investigation.

Objectives

- Analyze EV charging session data.
- Identify unusual charging patterns.
- Flag potential anomalies for review.
- Export results to Excel and create a summary chart.

Tools and Technologies

- Python
- Pandas
- Scikit-learn
- Matplotlib
- Excel
- Git and GitHub

Methodology

1. Load the EV charging dataset from Excel.
2. Clean the column names and prepare the data.
3. Filter out records with missing or invalid values in the selected features.
4. Apply the Isolation Forest anomaly detection algorithm.
5. Label sessions as "Normal" or "Potential Anomaly".
6. Export the results and create a chart summarizing the findings.

Features Used

- Session duration
- Energy consumed (kWh)
- Price per kWh
- Total cost

Outputs

- "anomaly_results.xlsx" — session-level analysis and anomaly labels.
- "anomaly_summary.png" — chart summarizing normal and potentially anomalous sessions.

Important Note

Potential anomalies are not confirmed cases of fraud or misuse. They require further investigation. The model's contamination parameter is set to 0.02 (2%), which influences the expected proportion of flagged records.

How to Run

1. Install the required libraries:
   
   python -m pip install pandas scikit-learn matplotlib openpyxl

2. Run the Python script:
   
   python anomaly_detection.py

Conclusion

This project demonstrates how machine learning can help identify unusual EV charging sessions and support further investigation into possible misuse or irregular activity.

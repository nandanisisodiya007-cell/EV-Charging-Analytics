# Task 15: Customer Lifetime Value (CLV) Estimation

## Project Overview
This project analyzes electric vehicle charging-session data to estimate customer value and identify high-value customer segments.

## Objectives
- Calculate total sessions and spending for each customer.
- Analyze customer energy consumption.
- Estimate annualized customer spending.
- Segment customers into Low, Medium, and High Value groups.
- Identify high-value customers for retention strategies.

## Tools Used
- Python
- Pandas
- Matplotlib
- Excel
- VS Code
- GitHub

## Methodology
1. Load charging-session records from the Sessions sheet.
2. Clean customer IDs, costs, energy values, and timestamps.
3. Aggregate session activity and spending by customer.
4. Calculate average spending per session and an annualized spending estimate.
5. Divide customers into three segments based on total spending.
6. Export customer results and a segment chart.

## Outputs
- results/customer_clv_results.csv
- results/customer_clv_results.xlsx
- results/clv_segment_chart.png

## Interpretation
Customers with higher total spending are classified as high-value customers. These segments can help prioritize customer engagement and retention efforts.

The annualized spending measure is an estimate based on observed activity, not a validated prediction of future lifetime value.

## How to Run
Install dependencies:

pip install pandas matplotlib openpyxl

Run the analysis:

python clv_analysis.py

## Limitations
The analysis uses historical charging activity and spending. It does not directly model future retention, profit margins, or customer churn.


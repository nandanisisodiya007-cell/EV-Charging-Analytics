import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

file_name = "task_4.xlsx"

df = pd.read_excel(file_name, sheet_name="Clustering")

# Take columns B:G
X = df.iloc[:, 1:7]

# Convert to numbers
X = X.apply(pd.to_numeric, errors="coerce")

# Keep rows with complete data
valid = X.notna().all(axis=1)

# Standardize 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X[valid])

# K-Means
kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

# Create cluster numbers
clusters = kmeans.fit_predict(X_scaled)

# Put clusters in column H
df.loc[valid, df.columns[7]] = clusters

# Show first 10 rows
print(df.iloc[:10, [0,7]])

# Save the clustering results to Excel
output = df.copy()

with pd.ExcelWriter(
    file_name,
    engine="openpyxl",
    mode="a",
    if_sheet_exists="replace"
) as writer:
    output.to_excel(writer,
sheet_name="Clustering", index=False)
    
    print ("Clustering sheet updated successfully!")
    


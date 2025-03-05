#%%
import polars as pl
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from pathlib import Path
import numpy as np

# define data paths
RAW_DATA_PATH = Path(__file__).parent.parent / "data"/ "raw"
INTERIM_DATA_PATH = Path(__file__).parent.parent / "data"/ "interim"
REPORTS_PATH = Path(__file__).parent.parent / "reports"/ "visualisations"
#%%
hmis_data = pl.read_csv(str(INTERIM_DATA_PATH) + "/hmis-filtered.csv")

# Convert Polars DataFrame to Pandas for sklearn compatibility
hmis_pandas = hmis_data.to_pandas()

# Define Features (Excluding Identifiers & Non-Numeric Columns)
indicator_columns = [col for col in hmis_pandas.columns if col not in [
    "id", "state_name", "district_name", "subdistrict_name", "sector", "date",
    "state_code", "district_code", "subdistrict_code"
]]
#%%
X = hmis_pandas[indicator_columns]

# Handle Missing Values with Median Imputation
imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X)

# Standardize Data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)
#%%
# Train Isolation Forest
iso_forest = IsolationForest(n_estimators=100, contamination='auto', random_state=42)
iso_forest.fit(X_scaled)

# Get Anomaly Scores as Pseudo Target Variable
anomaly_scores = iso_forest.decision_function(X_scaled)
y_pseudo = (anomaly_scores < np.percentile(anomaly_scores, 20)).astype(int)  # Mark bottom 10% as anomalies

# Train Random Forest with Pseudo Target
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_scaled, y_pseudo)

# Extract Feature Importances
feature_importance = pd.DataFrame({
    "Feature": indicator_columns,
    "Importance": rf_model.feature_importances_
}).sort_values(by="Importance", ascending=False)

# Display Top 20 Features
print(feature_importance.head(20))

# %%
import plotly.express as px

# Create interactive bar chart
fig = px.bar(
    feature_importance.head(20), 
    x="Importance", 
    y="Feature", 
    orientation="h", 
    title="Top 20 Important Features",
    labels={"Importance": "Feature Importance Score", "Feature": "Feature Name"},
    color="Importance",
    color_continuous_scale="viridis"
)

fig.update_layout(yaxis=dict(autorange="reversed"))  # Reverse Y-axis
fig.show()
# save the fig into reports path
fig.write_html(str(REPORTS_PATH) + "/feature_importance.html")

# %%
# save the top 20 selected features to a csv file
feature_importance.head(20).to_csv(str(INTERIM_DATA_PATH) + "/feature_importance.csv", index=False)
# %%

# %%
# import libraries
import polars as pl
from pathlib import Path
# let us plot using plotly
import plotly.express as px
import plotly.graph_objects as go


# define data paths
RAW_DATA_PATH = Path(__file__).parent.parent / "data"/ "raw"
INTERIM_DATA_PATH = Path(__file__).parent.parent / "data"/ "interim"
REPORTS_PATH = Path(__file__).parent.parent / "reports"/ "visualisations"
# %%
# read data/interim/major-health-indicators-subdistrict-level.csv using polars
hmis_data = pl.read_csv(str(INTERIM_DATA_PATH) + "/major-health-indicators-subdistrict-level.csv")
# check schema in polars
hmis_data.schema
# %%
# Convert date column to datetime type
hmis_data = hmis_data.with_columns(
    pl.col("date").str.strptime(pl.Datetime, "%Y-%m-%d")
)
# %%
# Fill missing numeric values with rolling mean (3-month window)
indicator_columns = [col for col in hmis_data.columns if col not in [
    "id","state_name", "district_name", "subdistrict_name", "sector", "date", 
    "state_code","district_code","subdistrict_code"]
    ]

# %%
# Calculate correlation matrix
correlation_matrix = hmis_data[indicator_columns].to_pandas().corr()
print(correlation_matrix)

# Identify highly correlated features (Pearson correlation > 0.9)
high_corr_pairs = [
    (col1, col2)
    for col1 in correlation_matrix.columns
    for col2 in correlation_matrix.columns
    if col1 != col2 and abs(correlation_matrix[col1][col2]) > 0.9
]
#%%
# Print highly correlated feature pairs
print(f"Highly correlated feature pairs: {high_corr_pairs}")
#%%
# Compute variance using Polars
variances = hmis_data.select(indicator_columns).var()
#%%
# Identify low variance features (threshold = 0.01)
low_variance_features = [col for col in variances.columns if variances[col][0] < 0.01]
#%%
# Print low variance features
print(f"Low variance features: {low_variance_features}")

#%%
# Drop redundant and low variance features
filtered_data = hmis_data.drop([col2 for _, col2 in high_corr_pairs] + low_variance_features)

#%%
# Print final number of features after filtering
print(f"Number of remaining features: {len(filtered_data.columns)}")

# save the filtered data to interim path
filtered_data.write_csv(str(INTERIM_DATA_PATH) + "/hmis-filtered.csv") 

# %%

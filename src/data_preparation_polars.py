#%%
# import libraries
import polars as pl
from pathlib import Path


# define data paths
RAW_DATA_PATH = Path(__file__).parent.parent / "data"/ "raw"
INTERIM_DATA_PATH = Path(__file__).parent.parent / "data"/ "interim"
REPORTS_PATH = Path(__file__).parent.parent / "reports"/ "visualisations"
# %%
# read data/raw/major-health-indicators-subdistrict-level.csv using polars
hmis_data = pl.read_csv(str(RAW_DATA_PATH) + "/major-health-indicators-subdistrict-level.csv")
# check schema in polars
hmis_data.schema
# %%
# describe the data
hmis_data.describe()
# keep only the rows where sector value is "Total"
hmis_data = hmis_data.filter(hmis_data["sector"] == "Total")
# %%
# total number of records in the data now
hmis_data.shape
# %%
# get the unique dates from hmis_data orderd chronologically
unique_dates = hmis_data.select("date").unique().sort("date")
print(unique_dates)


# %%
# Ensure date column is in the correct format (assuming dd-mm-yyyy format)
# Convert the date column to the correct format
hmis_data = hmis_data.with_columns(
    pl.col("date").str.to_date("%Y-%m-%d").alias("date")
)

# Verify the conversion
print(hmis_data["date"].head())

# %%
# Create the full expected date range from the earliest to the latest date
full_date_range = (
    pl.date_range(start=hmis_data["date"].min(), end=hmis_data["date"].max(), interval="1mo")
    .to_frame("date")
)

# Identify missing months by performing an anti-join
missing_dates = full_date_range.join(hmis_data, on="date", how="anti")

print(missing_dates)

# %%
full_date_range = pl.DataFrame(
    {
        "date": pl.date_range(
            start=hmis_data["date"].min(),
            end=hmis_data["date"].max(),
            interval="1mo",
            eager=True  # Ensures the result is immediately evaluated
        )
    }
)

# %%
missing_dates = full_date_range.join(hmis_data, on="date", how="anti")

print(missing_dates)
#%%
# Add all columns to missing_data with default values
for col in hmis_data.columns:
    if col not in missing_dates.columns:
        missing_dates = missing_dates.with_columns(pl.lit(None).alias(col))

#%%
missing_data = missing_dates.select(hmis_data.columns)

# %%
# Get unique subdistrict-related identifiers
unique_subdistricts = hmis_data.select([
    "state_name", "state_code", "district_name", "district_code", 
    "subdistrict_name", "subdistrict_code"
]).unique()

# Perform cross join and select only the necessary columns
missing_data = unique_subdistricts.join(missing_dates, how="cross").select(
    unique_subdistricts.columns + ["date"]
)

# Ensure missing data has all required columns with NaN for missing values
for col in hmis_data.columns:
    if col not in missing_data.columns:
        missing_data = missing_data.with_columns(pl.lit(None).alias(col))

# Align column order with original data
missing_data = missing_data.select(hmis_data.columns)

# Concatenate original data with filled missing data and remove duplicates
full_data = pl.concat([hmis_data, missing_data]).unique().sort(["subdistrict_code", "date"])

# Confirm final shape
print(full_data.shape)

# %%
# Fill missing numeric values with rolling mean (3-month window)
indicator_columns = [col for col in full_data.columns if col not in ["state_name", "district_name", "subdistrict_name", "sector", "date", "state_code","district_code","subdistrict_code"]]

# Apply forward fill first to propagate known values
full_data = full_data.with_columns([
    pl.col(col).fill_null(strategy="forward").fill_null(strategy="backward").alias(col)
    for col in indicator_columns
])

# Then apply rolling mean within each subdistrict
full_data = full_data.with_columns([
    pl.col(col).rolling_mean(window_size=3).over("subdistrict_code").alias(col)
    for col in indicator_columns
])

# Handle remaining nulls (if any persist after rolling mean)
full_data = full_data.with_columns([
    pl.col(col).fill_null(strategy="mean").alias(col)
    for col in indicator_columns
])

# Check results
print(full_data.head())

# %%
# write data/interim/major-health-indicators-subdistrict-level.csv
full_data.write_csv(str(INTERIM_DATA_PATH) + "/major-health-indicators-subdistrict-level.csv")

# %%

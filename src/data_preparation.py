#%%
# libraries pyspark
import pandas as pd
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import pyspark.sql.types as T
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql.functions import concat, lit
from pyspark.sql.functions import year, month
from pyspark.sql.functions import when
from pyspark.ml.feature import VectorAssembler
from sklearn.ensemble import IsolationForest
from pyspark.sql.window import Window
from pyspark.sql.functions import lag
from pyspark.sql.functions import avg, stddev
from pyspark.sql import functions as F
from pyspark.sql.functions import col, sum
from pyspark.sql.types import DateType
from datetime import datetime


# define data paths
RAW_DATA_PATH = Path(__file__).parent.parent / "data"/ "raw"
INTERIM_DATA_PATH = Path(__file__).parent.parent / "data"/ "interim"
REPORTS_PATH = Path(__file__).parent.parent / "reports"/ "visualisations"
#%%
# read data/major-health-indicators-subdistrict-level.csv using spark
# spark = SparkSession.builder.appName("hmis").getOrCreate()

# Increase Spark driver memory
spark = SparkSession.builder \
    .appName("HMIS Anomaly Detection") \
    .config("spark.driver.memory", "16g") \
    .getOrCreate()

#%%
# hmis_data = spark.read.csv(str(RAW_DATA_PATH) + "/major-health-indicators-subdistrict-level.csv", header=True, inferSchema=True)
# Read CSV with explicit date parsing
hmis_data = (
    spark.read.option("header", "true")
    .option("inferSchema", "true")
    .csv(str(RAW_DATA_PATH) + "/major-health-indicators-subdistrict-level.csv")
    .withColumn("date", F.to_date(F.col("date"), "dd-MM-yyyy"))
)
# check schema
hmis_data.printSchema()

#%%
# describe the data
hmis_data.describe().show()
# keep only the rows where sector value is "Total"
hmis_data = hmis_data.filter(hmis_data["sector"] == "Total")
# %%
# Step 1: Identify the unique identifier columns for subdistricts
unique_identifiers = hmis_data.select(
    "state_name", "state_code", "district_name", "district_code", 
    "subdistrict_name", "subdistrict_code","sector"
).distinct()

# Step 2: Create a DataFrame for the missing months (April and May of 2017)
missing_dates = spark.createDataFrame(
    [("2017-04-01",), ("2017-05-01",)],
    ["date"]
).withColumn("date", F.to_date(F.col("date")))

# Step 3: Cross join unique identifiers with missing dates
missing_data = unique_identifiers.crossJoin(missing_dates)

# Step 4: Merge with the original data, making sure not to overwrite existing data
# Union the original data with the missing rows and keep distinct records
full_data = hmis_data.unionByName(missing_data, allowMissingColumns=True).distinct()

# Step 5: Sort by subdistrict_code and date to ensure ordered data
full_data = full_data.orderBy("subdistrict_code", "date")

# Check the final dataset count and ensure dates have been added
print(f"Total records after adding missing months: {full_data.count()}")

# Display a sample of the final dataset to verify
full_data.show(10)

#%%
# show data for 01-01-2018 from full_data
full_data.filter(full_data["date"] == "2017-04-01").show()
# %%
# Add year and month columns for easier grouping
full_data = full_data.withColumn("year", year("date")).withColumn("month", month("date"))
# Aggregate indicators by month and calculate averages oreder by year and month
monthly_trends = full_data.groupBy("year", "month").mean()
# monthly_trends.orderBy("year", "month").show()
# ignore id, state_code, district_code, subdistrict_code
monthly_trends = monthly_trends.drop("avg(id)", "avg(state_code)", "avg(district_code)", "avg(subdistrict_code)")
monthly_trends.orderBy("year", "month").show()

# %%
# build plots line plots for each indicator in monthly_trends and save the plots in reports/visualisations folder

# Create a "year-month" column for easier plotting
monthly_trends = monthly_trends.withColumn("year_month", concat(monthly_trends["year"].cast("string"), lit("-"), monthly_trends["month"].cast("string")))

# Convert Spark DataFrame to pandas for plotting
monthly_trends_pd = monthly_trends.orderBy("year", "month").toPandas()

# Use coalesce(1) to combine all data into a single partition
monthly_trends.orderBy("year", "month") \
    .coalesce(1) \
    .write.csv(str(INTERIM_DATA_PATH) + "/monthly_trends", header=True, mode="overwrite")

# %%    

# Plot each indicator over time
for indicator in monthly_trends.columns[3:]:
    plt.figure(figsize=(10, 5))
    sns.lineplot(x="year_month", y=indicator, data=monthly_trends_pd)
    plt.title(f"{indicator} over Time")
    plt.xlabel("Year-Month")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(str(REPORTS_PATH) + f"/{indicator}.png")
    plt.close()

# %%
# Define COVID period as January to September 2020
full_data = full_data.withColumn(
    "is_covid_period",
    when((full_data["year"] == 2020) & (full_data["month"].between(1, 9)), 1).otherwise(0)
)
#%%

# %%
# Define feature columns: All columns except identifiers and non-numeric columns
feature_columns = [
    col for col in hmis_data.columns 
    if col not in ["date", "state_name", "state_code", "district_name", 
                   "district_code", "subdistrict_name", "subdistrict_code", 
                   "sector", "year", "month", "is_covid_period"]
]
columns_for_analysis = feature_columns + ["date", "year", "month", "is_covid_period"]
#%%
# 3-Month Rolling Mean Imputation
# Define a window partitioned by subdistrict and ordered by date for calculating rolling mean
window_spec = Window.partitionBy("subdistrict_code").orderBy("date").rowsBetween(-2, 0)  # 3-month rolling window

# Apply rolling mean to each numeric column for imputation
numeric_columns = [col[0] for col in full_data.dtypes if col[1] in ("int", "double", "float")]

for col in numeric_columns:
    full_data = full_data.withColumn(
        f"{col}_imputed",
        F.when(F.col(col).isNull(), F.avg(F.col(col)).over(window_spec)).otherwise(F.col(col))
    )

# Drop the original columns if necessary and rename imputed columns back to the original names
for col in numeric_columns:
    full_data = full_data.drop(col).withColumnRenamed(f"{col}_imputed", col)

# Check results
# full_data.show()
#%%
# Enable checkpointing
spark.sparkContext.setCheckpointDir("/tmp/spark_checkpoints")

# Perform a portion of the transformation and checkpoint
intermediate_data = full_data.checkpoint(eager=True)
# Cache the intermediate result
intermediate_data = full_data.cache()
#%%
# Define window to partition by sub-district and order by date
window_spec = Window.partitionBy("subdistrict_name").orderBy("date")

# Create 1-month and 3-month lags for each feature
for column in feature_columns:
    hmis_data = hmis_data.withColumn(f"{column}_lag1", lag(column, 1).over(window_spec))
    hmis_data = hmis_data.withColumn(f"{column}_lag3", lag(column, 3).over(window_spec))
# %%
# Calculate rolling averages and standard deviations over a 3-month window
for column in feature_columns:
    hmis_data = hmis_data.withColumn(f"{column}_roll_avg3", avg(column).over(Window.partitionBy("subdistrict_name").orderBy("date").rowsBetween(-3, 0)))
    hmis_data = hmis_data.withColumn(f"{column}_roll_std3", stddev(column).over(Window.partitionBy("subdistrict_name").orderBy("date").rowsBetween(-3, 0)))
# %%
# Calculate average values for each indicator at the district level
for column in feature_columns:
    district_avg_col = f"{column}_district_avg"
    hmis_data = hmis_data.withColumn(district_avg_col, avg(column).over(Window.partitionBy("district_name")))
#%%
# Split data into pre-COVID and post-COVID periods
pre_covid_data_pd = hmis_data.filter((hmis_data["year"] < 2020) | ((hmis_data["year"] == 2020) & (hmis_data["month"] < 1)))
post_covid_data_pd = hmis_data.filter((hmis_data["year"] > 2020) | ((hmis_data["year"] == 2020) & (hmis_data["month"] > 9)))

# %%
# Define feature columns: All columns except identifiers and non-numeric columns
feature_columns = [
    col for col in hmis_data.columns 
    if col not in ["date", "state_name", "state_code", "district_name", 
                   "district_code", "subdistrict_name", "subdistrict_code", 
                   "sector", "year", "month", "is_covid_period"]
]
columns_for_analysis = feature_columns + ["date", "year", "month", "is_covid_period"]
# %%
# Define window to partition by sub-district and order by date
window_spec = Window.partitionBy("subdistrict_name").orderBy("date")

# Create 1-month and 3-month lags for each feature
for column in feature_columns:
    hmis_data = hmis_data.withColumn(f"{column}_lag1", lag(column, 1).over(window_spec))
    hmis_data = hmis_data.withColumn(f"{column}_lag3", lag(column, 3).over(window_spec))
# %%
# Calculate rolling averages and standard deviations over a 3-month window
for column in feature_columns:
    hmis_data = hmis_data.withColumn(f"{column}_roll_avg3", avg(column).over(Window.partitionBy("subdistrict_name").orderBy("date").rowsBetween(-3, 0)))
    hmis_data = hmis_data.withColumn(f"{column}_roll_std3", stddev(column).over(Window.partitionBy("subdistrict_name").orderBy("date").rowsBetween(-3, 0)))


# %%
# Calculate average values for each indicator at the district level
for column in feature_columns:
    district_avg_col = f"{column}_district_avg"
    hmis_data = hmis_data.withColumn(district_avg_col, avg(column).over(Window.partitionBy("district_name")))

# %%
# Filter out rows with any missing values in new features
hmis_data = hmis_data.na.drop()

# %%
# Convert to pandas for training, excluding date-related columns
columns_for_model = feature_columns + [f"{col}_lag1" for col in feature_columns] + \
                    [f"{col}_lag3" for col in feature_columns] + \
                    [f"{col}_roll_avg3" for col in feature_columns] + \
                    [f"{col}_roll_std3" for col in feature_columns] + \
                    [f"{col}_district_avg" for col in feature_columns]

hmis_data_pd = hmis_data.select(columns_for_model).toPandas()

# %%
# Get all available columns in hmis_data
available_columns = hmis_data.columns

# Keep only columns from columns_for_model that exist in hmis_data
columns_for_model = [col for col in columns_for_model if col in available_columns]


# %%
# Add identifier columns to columns_for_model if they exist
identifiers = ["date", "year", "month", "subdistrict_name"]
columns_for_model.extend([id_col for id_col in identifiers if id_col in available_columns])

# %%
# Convert to pandas for training, ensuring we include identifier columns
hmis_data_pd = hmis_data.select(columns_for_model).toPandas()

# %%
# Add `is_covid_period` for context after model predictions
columns_for_model.append("is_covid_period")


# %%
# Convert to pandas, including only columns for model plus contextual columns
context_columns = ["state_name", "district_code", "state_code", "sector", "district_name", "subdistrict_code"]
columns_for_pandas = columns_for_model + context_columns

# Convert Spark DataFrame to pandas DataFrame for model training
hmis_data_pd = hmis_data.select(columns_for_pandas).toPandas()

# %%
# Save the required columns to a CSV file
hmis_data.select(columns_for_pandas).write.csv(str(INTERIM_DATA_PATH) + "/"+"hmis_data", header=True, mode="overwrite")


# %%
# Select only required columns for the CSV export
simple_hmis_data = hmis_data.select(*columns_for_pandas)
# Persist to reduce the load of complex transformations
simple_hmis_data.cache()
# Write to Parquet as an intermediary step
simple_hmis_data.write.parquet(str(INTERIM_DATA_PATH) + "/"+"hmis_data_parquet", mode="overwrite")

# Then load and save as CSV, or convert from Parquet to CSV using an external tool
simple_hmis_data_parquet = spark.read.parquet(str(INTERIM_DATA_PATH) + "/"+"hmis_data_parquet")
simple_hmis_data_parquet.write.csv(str(INTERIM_DATA_PATH) + "/"+"hmis_data", header=True, mode="overwrite")
# %%

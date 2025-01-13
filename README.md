# Anomaly Detection on Health Major Indicators

## Introduction

The goal of this project is to build anamoly detection model for early warning system to predict the severity of a health conditions based on the health major indicators.

References:
1. https://medium.com/towards-data-science/ensemble-learning-for-anomaly-detection-955efb1b2fac
2. https://medium.com/@corymaklin/isolation-forest-799fceacdda4
3. https://towardsdatascience.com/anamoly-detection-techniques-to-detect-outliers-fea92047a222

Source: Health Management Information System
Time Period: 2017-2021
Frequency: Monthly
Indicators: 286

## Prerequisites
Data from HMIS to be prepared as is using idp-datasets scripts and existing dataset on [India Data Portal](https://indiadataportal.com/p/health-management-information-system-hmis/r/mohfw-hmis-sd-mn-fhi) with LGD Mapping completed (90% through scripts and rest manually). Need to verify LGD Mapping according to the latest LGD Directory.

# Project Documentation: Health Indicator Anomaly Detection for Early Warning System

## Project Overview

The purpose of this project is to develop an anomaly detection system for monitoring health indicators at the subdistrict level, with a monthly time frequency. The model will serve as an early warning system, enabling health administrators to respond proactively to unusual health trends across various demographic and health categories.

### Data Overview

- **Source**: Health Management Information System (HMIS), India.
- **Time Period**: 2017-2021 (monthly data).
- **Indicators**: 286 health indicators.
- **Granularity**: Subdistrict-level data.
- **Rows**: ~547,000 records.

## Project Goals

1. **Data Preparation**: Load, clean, and transform data to make it suitable for time-series anomaly detection.
2. **Feature Engineering**: Create lagged, rolling average, and statistical features to enhance anomaly detection models.
3. **Model Exploration**: Experiment with various anomaly detection approaches to determine the best model(s) for identifying unusual health trends.

---

## Steps Completed So Far

### Step 1: Data Loading and Initial Exploration

1. **Data File**: Loaded the primary dataset and explored it for structure, frequency, and granularity.
2. **Metadata**: The data was accompanied by a codebook with detailed metadata for each indicator. This helped classify variables by demographic categories (e.g., children, women, general).

### Step 2: Dataset Splitting by Demographics

To manage the high-dimensional data and improve interpretability, the indicators were split into three subcategories based on demographics:
   - **Children-related indicators**
   - **Women-related indicators**
   - **General health indicators**

This separation was aimed at reducing data size for individual models and optimizing feature engineering for specific demographic patterns.

### Step 3: General Data Preparation

1. **Core Identifier Columns**: Ensured that identifier columns such as `date`, `state`, `district`, `subdistrict`, and `sector` were retained for all records.
2. **Feature Columns**: Selected indicator columns to include in the model after excluding non-numeric and irrelevant columns.
3. **Pre- and Post-COVID Split**: Given that COVID-19 significantly impacted health trends, we split the dataset into pre-COVID and post-COVID segments for separate modeling.

### Step 4: Data Continuity Checks

To ensure the consistency and reliability of time-series data, we conducted continuity checks to identify missing months in each subdistrict and ensure that our time series analysis accurately reflects health trends.

1. **Identify Missing Months**: 
   - Grouped data by subdistrict and month-year to detect any missing monthly records.
   - **Observation**: Identified some missing months, notably in 2017.

2. **Impute Missing Months**:
   - Options considered include forward-fill, backward-fill, or seasonal interpolation based on the nature of missing data and continuity requirements.

3. **Implementation Steps**:
   - **Identify Missing Months**: Generated a list of missing months for review.
   - **Impute Missing Data**: Chose imputation methods based on the extent and distribution of missing values.

### Step 5: Feature Engineering

To capture temporal dependencies in the data, we engineered several types of features:

1. **Lag Features**: Created lagged versions of each indicator for 1-month and 3-month intervals.
2. **Rolling Averages and Standard Deviations**: Calculated 3-month rolling averages and standard deviations for each indicator.
3. **District-Level Averages**: Computed average values for each indicator at the district level to account for geographical trends.

These features provided richer temporal and spatial information for anomaly detection models.

### Step 6: Initial Anomaly Detection Modeling

We explored **Isolation Forest** as an initial approach, training separate models on the pre-COVID and post-COVID datasets to capture distinct patterns influenced by the pandemic.

### Step 7: Model Challenges and Observations

- **Data Dimensionality**: With over 1,700 feature columns (from original features and engineered ones) and 547,000 rows, computational load and memory management became significant concerns.
- **Error During Pandas Conversion**: Attempting to convert the Spark DataFrame to a Pandas DataFrame for further processing or exporting to CSV led to memory errors and slow processing times.

---

## Model Exploration

To identify the best approach for anomaly detection, we are considering multiple models and methodologies:

1. **Isolation Forest**: An unsupervised model focusing on identifying data outliers.
2. **Time-Series Anomaly Detection Models**:
   - **LSTM (Long Short-Term Memory)**: A neural network model effective in handling sequential dependencies.
   - **SARIMA (Seasonal Autoregressive Integrated Moving Average)**: A statistical model suited for univariate time series with seasonality.
3. **Hybrid Approaches**: Combining traditional and deep learning methods to capture both local anomalies and complex sequential patterns.

These models will be evaluated based on their ability to detect health anomalies effectively while managing computational efficiency.

---

## Next Steps and Options for Scaling

1. **Evaluate Alternatives for Large Data Handling**: Given the size and complexity of the dataset, consider options like Apache Hudi, Delta Lake, Dask, or cloud databases optimized for large datasets.
2. **Storage Optimization**: Assess storage formats like Parquet or Avro for efficient I/O performance.
3. **Distributed Computing Framework**: Re-evaluate distributed frameworks for preprocessing and model training, potentially using AWS Glue, GCP Dataflow, or Databricks.

---

## Summary

This project has reached a point where data storage, I/O optimization, and processing framework decisions are crucial to ensure scalability. Our engineered features and model approach are designed to capture complex patterns, but a robust infrastructure is required to support the model experiments and handle the data volume.

---



---






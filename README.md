# HMIS Anomaly Detection Project

## 1. Overview
### 1.1 What are we trying to achieve?
The goal of this project is to develop an anomaly detection system for health indicators at the subdistrict level using time-series data from the Health Management Information System (HMIS), India. By identifying deviations from normal health trends, the model aims to serve as an early warning system for public health anomalies.

## 2. Understanding the Data
- **Source**: Health Management Information System (HMIS), India
- **Time Period**: 2017-2021 (monthly data)
- **Indicators**: 286 health-related indicators
- **Granularity**: Subdistrict-level data
- **Challenges**:
  - Missing months in 2017 (April, May)
  - High dimensionality (originally 1,700+ columns due to feature engineering)
  - Duplicate and redundant records (sector-wise categorization)
  
## 3. Data Preparation
- **Filtering Sector Data**: Only ‘Total’ sector records were retained to eliminate redundancy.
- **Handling Missing Dates**: Missing months were imputed using backward fill and rolling mean techniques.
- **Ensuring Data Continuity**: Cross-joining unique subdistricts with the full date range ensured a consistent time-series structure.
- **Data Type Standardization**: Subdistrict codes were stored as strings but represented as numeric values to maintain consistency.

## 4. Exploratory Data Analysis (EDA)
We tried to visualise each indicator over time-period across random sub districts to see if the data shows any seasonality or patterns. It turns out there are no such pattersn if we observe randomly.
### 4.1 Removal of High Correlated and Low Variance Features
- **High Correlation**: Features with Pearson correlation > 0.9 were removed to prevent redundancy.
- **Low Variance**: Features with near-zero variance were dropped, as they do not contribute to anomaly detection.
- **Final Feature Set**: After filtering, 211 indicators remained for further analysis.

## 5. Isolation Forest and Random Forest for Important Feature Selection (Why Not PCA?)
- **Isolation Forest**: Used to detect anomalies by identifying points with short average path lengths in decision trees.
- **Random Forest Feature Importance**:
  - Assigned importance scores to features based on their contribution to anomaly detection.
  - Used permutation importance since Isolation Forest does not provide built-in feature importances.
- **Why Not PCA?**:
  - PCA reduces dimensionality but does not preserve interpretability of individual features.
  - Health administrators need insights on which specific indicators are anomalous.
  - PCA assumes linearity, which may not hold for complex health trends.
  
## 6. LSTM Model and Why This Was Chosen
- **Why LSTM?**
  - Captures sequential dependencies and long-term temporal patterns.
  - Effective for modeling health trends where past observations influence future patterns.
  - Can learn non-linear relationships between features.
- **Data Splitting Strategy**:
  - **Training (2017-2019)**: Model learns historical patterns.
  - **Validation (2020)**: Evaluates generalization to newer data.
  - **Testing (2021)**: Final evaluation on the most recent trends.
- **LSTM Autoencoder for Anomaly Detection**:
  - Trained to reconstruct normal sequences; high reconstruction error signals anomalies.
  - Outputs an anomaly score based on deviation from expected patterns.

This structured pipeline ensures robust anomaly detection while maintaining interpretability, crucial for public health monitoring.


---






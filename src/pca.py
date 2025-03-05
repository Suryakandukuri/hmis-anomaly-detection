#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
# %%
hmis_data = pd.read_csv("../data/interim/major-health-indicators-subdistrict-level.csv")
hmis_data.head()
# %%
hmis_data.dtypes
# %%
hmis_data["date"] = pd.to_datetime(hmis_data["date"])
# %%
feature_columns = [
    col for col in hmis_data.columns 
    if col not in ["id", "date", "state_name", "state_code", "district_name", 
                   "district_code", "subdistrict_name", "subdistrict_code", 
                   "sector", "year", "month", "is_covid_period"]
]
# %%
scaler = StandardScaler()
#scale the data on the features columns using StandardScaler
scaled_data = scaler.fit_transform(hmis_data[feature_columns])
# %%

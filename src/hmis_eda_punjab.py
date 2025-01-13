# libraries
import pandas as pd
from ydata_profiling import ProfileReport

# read data/punab-major-health-indicators-subdistrict-level.xlsx
df = pd.read_excel("data/punab-major-health-indicators-subdistrict-level.xlsx")
profile = ProfileReport(df)
profile.to_file("hmis_eda_punjab.html")
import sys
sys.path.insert(1, '/Users/nicolaschapman/Documents/DigitalTwinCattle/DigitalTwinCattle/TrendsAndFiltering')
sys.path.insert(1, '/Users/nicolaschapman/Documents/DigitalTwinCattle/DigitalTwinCattle/Regressions')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
from statsmodels.tsa.ar_model import AutoReg
from filter_data import *
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from statsmodels.tsa.stattools import grangercausalitytests
import warnings
import math
import random
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# import data
panting_df = pd.read_csv("Clean IIR Output/panting_timeseries.csv")
resting_df = pd.read_csv("Clean IIR Output/resting_timeseries.csv")
eating_df = pd.read_csv("Clean IIR Output/eating_timeseries.csv")
rumination_df = pd.read_csv("Clean IIR Output/rumination_timeseries.csv")
medium_activity_df = pd.read_csv("Clean IIR Output/medium activity_timeseries.csv")
weather_df = pd.read_csv("Weather.csv", encoding='utf-8')
weather_df = weather_df.iloc[[i for i in range(288,9163,6)]+[j for j in range(9164,10791,6)]]

# extract ordered list of animals and data type
cow_list = list(set(panting_df["Cow"]))
cow_list = sorted(cow_list)
data_type_list = list(set(panting_df["Data Type"]))
data_type_list = sorted(data_type_list)

# extract series
HLI_time_series = weather_df["HLI"].values
# extract all cow series
panting_sequences = []
for cow in cow_list:
    # if cow=='All':
    #     continue
    example_panting = panting_df[(panting_df["Cow"]=="8021870")&(panting_df["Data Type"]=="panting filtered")].values[0][2:]
    panting_sequences.append(example_panting)
# calculate relative
relative_raw_series = [a/b for a, b in zip(example_panting, HLI_time_series)]

# normalised data
scalar_pant = StandardScaler()
scalar_pant.fit(np.array(panting_sequences).reshape(-1,1))
panting_sequences_norm = []
for series in panting_sequences:
    panting_sequences_norm.append(scalar_pant.transform(np.array(series).reshape(-1,1)))

scalar_HLI = StandardScaler()
HLI_series_norm = scalar_HLI.fit_transform(np.array(HLI_time_series).reshape(-1,1))
relative_norm_series = [a - b for a, b in zip(panting_sequences_norm[0], HLI_series_norm)]


fig, ax = plt.subplots()
ax.plot(HLI_time_series)
ax.set_ylabel("weather",fontsize=14)
ax2=ax.twinx()
ax2.plot(example_panting, 'r-')
ax2.set_ylabel("panting",fontsize=14)

fig, ax = plt.subplots()
ax.plot(relative_raw_series)
ax.set_ylabel("relative",fontsize=14)
ax2=ax.twinx()
ax2.plot(example_panting, 'r-')
ax2.set_ylabel("panting",fontsize=14)

fig, ax = plt.subplots()
ax.plot(HLI_series_norm)
ax.set_ylabel("weather",fontsize=14)
ax2=ax.twinx()
ax2.plot(panting_sequences_norm[0], 'r-')
ax2.set_ylabel("panting",fontsize=14)

fig, ax = plt.subplots()
ax.plot(relative_norm_series)
ax.set_ylabel("relative",fontsize=14)
ax2=ax.twinx()
ax2.plot(panting_sequences_norm[0], 'r-')
ax2.set_ylabel("panting",fontsize=14)

plt.show()


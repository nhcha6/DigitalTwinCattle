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


def generate_model_data(df, state, cows):
    print(state)
    new_filtered_data = []
    for cow in cows:
        print(cow)
        raw_df = df[(df["Cow"] == cow) & (df["Data Type"] == state)]
        filtered_df = df[(df["Cow"] == cow) & (df["Data Type"] == state.replace("raw", "filtered"))]
        for i in range(201, 1705):
            new_data = [cow, i]
            raw_data = raw_df[[str(j) for j in range(i - 200, i)]].values[0]
            filt_data = filtered_df[[str(j) for j in range(i, i + 48)]].values[0]
            filt_seq = butter_lp_filter([cutoff_dict[state]], [4], raw_data, "All", False)

            new_data.extend(filt_seq)
            new_data.extend(filt_data)
            new_filtered_data.append(new_data)

    column_headers = ['Cow', 'next_ts']
    column_headers += [x for x in range(-200, 48)]
    model_data_df = pd.DataFrame(new_filtered_data, columns=column_headers)
    # model_data_df.to_csv("Model Data/" + state[0:-4] + " model data.csv", index = False)
    model_data_df.to_pickle("Model Data/" + state[0:-4] + " model data.pkl")

# import data
panting_df = pd.read_csv("Clean Dataset Output/panting_timeseries.csv")
resting_df = pd.read_csv("Clean Dataset Output/resting_timeseries.csv")
eating_df = pd.read_csv("Clean Dataset Output/eating_timeseries.csv")
rumination_df = pd.read_csv("Clean Dataset Output/rumination_timeseries.csv")
medium_activity_df = pd.read_csv("Clean Dataset Output/medium activity_timeseries.csv")
weather_df = pd.read_csv("Weather.csv", encoding='utf-8')
weather_df = weather_df.iloc[[i for i in range(288,9163,6)]+[j for j in range(9164,10791,6)]]

# extract ordered list of animals and data type
cow_list = list(set(panting_df["Cow"]))
cow_list = sorted(cow_list)
data_type_list = list(set(panting_df["Data Type"]))
data_type_list = sorted(data_type_list)

cutoff_dict = {'panting raw': 3, 'resting raw': 4.5, 'rumination raw': 4.5, 'medium activity raw': 4, 'eating raw': 4}

# panting_model_data = pd.read_pickle("Model Data/panting model data.pkl")
# print(panting_model_data.head(20))

generate_model_data(resting_df, "resting raw", cow_list)
generate_model_data(resting_df, "resting raw", cow_list)
generate_model_data(medium_activity_df, "medium activity raw", cow_list)

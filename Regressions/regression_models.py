import sys
sys.path.insert(1, '/Users/nicolaschapman/Documents/DigitalTwinCattle/DigitalTwinCattle/TrendsAndFiltering')

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from filter_data import *
from pandas.plotting import autocorrelation_plot

def build_prev_x_data(lower, upper, rows):
    x = []
    y = []

    horizon = 1
    lag = 26

    for i in range(lower,upper,1):
        # extract section of time series to analyse
        [raw_ts, filtered_ts] = rows[[str(j) for j in range(i, i + lag)]].values.tolist()

        # filter data
        filtered_raw = butter_lp_filter([3], [4], raw_ts, "All", False)

        # apply differencing
        filtered_season_diff = diff(filtered_raw, 1)
        raw_season_diff = diff(raw_ts, 1)
        filtered_double_diff = diff(filtered_season_diff, 1)
        #raw_double_diff = diff(raw_season_diff, 1)

        # declare x_data
        # original time-series
        # x_data = raw_ts + filtered_ts
        # seasonally differenced
        #x_data = filtered_season_diff# + raw_season_diff
        # double differenced
        x_data = filtered_double_diff#+ raw_double_diff

        # extract y-data
        # original data
        #y_data = filtered_ts[i+horizon+lag]
        # seasonal_diff
        y_data = rows.iloc[1][str(i + horizon + lag - 1)] - rows.iloc[1][str(i + horizon + lag - 2)]
        # double diff
        y_prev = rows.iloc[1][str(i + horizon + lag - 2)] - rows.iloc[1][str(i + horizon + lag - 3)]
        y_data = y_data - y_prev

        # plot some data for testing
        # plt.plot(filtered_raw[24:])
        # plt.plot(filtered_season_diff)
        # plt.plot(raw_ts[24:])
        # plt.plot(raw_season_diff)
        # plt.show()

        x.append(x_data)
        y.append(y_data)

    return x, y

def build_prev_15_data(lower, upper, rows):
    x = []
    y = []

    horizon = 1
    # 15 days lag to midday
    lag = 372

    for i in range(lower,upper,1):
        # extract section of time series to analyse
        [raw_ts, filtered_ts] = rows[[str(j) for j in range(i, i + lag)]].values.tolist()

        # filter data
        filtered_raw = butter_lp_filter([3], [4], raw_ts, "All", False)

        # apply differencing
        filtered_season_diff = diff(filtered_raw, 24)
        raw_season_diff = diff(raw_ts, 24)
        filtered_double_diff = diff(filtered_season_diff, 1)
        raw_double_diff = diff(raw_season_diff, 1)

        # declare x_data
        # seasonally differenced
        x_data = [filtered_season_diff[i] for i in range(11+horizon, len(filtered_season_diff),24)]
        x_data += [raw_season_diff[i] for i in range(11+horizon, len(filtered_season_diff),24)]
        # double differenced
        # x_data = filtered_double_diff+ raw_double_diff

        # extract y-data
        # original data
        #y_data = filtered_ts[i+horizon+lag]
        # seasonal_diff
        y_data = rows.iloc[1][str(i + horizon + lag - 1)] - rows.iloc[1][str(i + horizon + lag - 25)]
        # double diff
        # y_prev = rows.iloc[1][str(i + horizon + lag - 2)] - rows.iloc[1][str(i + horizon + lag - 26)]
        # y_data = y_data - y_prev

        # plot some data for testing
        # plt.plot(filtered_raw[24:])
        # plt.plot(filtered_season_diff)
        # plt.plot(raw_ts[24:])
        # plt.plot(raw_season_diff)
        # plt.show()

        # if np.isnan(x_data).any():
        #     print("invalid data for " + cow)
        #     break

        x.append(x_data)
        y.append(y_data)

    return x, y

def build_combined_data(lower, upper, rows):
    x = []
    y = []

    horizon = 1
    lag = 360

    for i in range(lower,upper,1):
        # extract section of time series to analyse
        [raw_ts, filtered_ts] = rows[[str(j) for j in range(i, i + lag)]].values.tolist()

        # filter data
        filtered_raw = butter_lp_filter([3], [4], raw_ts, "All", False)

        # apply differencing
        filtered_season_diff = diff(filtered_raw, 1)
        #raw_season_diff = diff(raw_ts, 1)
        filtered_double_diff = diff(filtered_season_diff, 1)
        #raw_double_diff = diff(raw_season_diff, 1)

        # declare x_data
        # index of lags to keep: 24 most recent and then every 24th from there
        #x_data = filtered_double_diff[-24:] + [filtered_double_diff[j] for j in range(22,lag-26,24)]
        x_data = [filtered_double_diff[j] for j in range(22,lag-25,24)]

        # extract y-data
        # original data
        #y_data = filtered_ts[i+horizon+lag]
        # seasonal_diff
        y_data = rows.iloc[1][str(i + horizon + lag - 1)] - rows.iloc[1][str(i + horizon + lag - 2)]
        # double diff
        y_prev = rows.iloc[1][str(i + horizon + lag - 2)] - rows.iloc[1][str(i + horizon + lag - 3)]
        y_data = y_data - y_prev

        # plot some data for testing
        # plt.plot(filtered_raw[24:])
        # plt.plot(filtered_season_diff)
        # plt.plot(raw_ts[24:])
        # plt.plot(raw_season_diff)
        # plt.show()

        x.append(x_data)
        y.append(y_data)

    return x, y

def diff(series, lag):
    log_series=series
    diff = [(log_series[j]-log_series[j-lag]) for j in range(lag,len(log_series))]
    return diff

df_panting = pd.read_csv("Clean Dataset Output/panting_timeseries.csv")

# ensure iterates through data in a consistent manner
cow_list = list(set(df_panting["Cow"]))
cow_list = sorted(cow_list)
#cow_list = ['8027107', '8022092', '8027476', '8032505', '8045911']
#cow_list = ['8027107']
data_type_list = list(set(df_panting["Data Type"]))
data_type_list = sorted(data_type_list)

# extract herd data
herd_rows = df_panting.loc[(df_panting["Cow"] == "All")]

# iterate through each animal and data type to build input data
x = []
y = []
x_test = []
y_test = []
counter = 0
for cow in cow_list:
    counter+=1
    print(counter)

    #plot data alterations
    # if counter in [1]:
    #     plt.figure()
    #     test_rows = df_panting.loc[(df_panting["Cow"] == cow)]
    #     [raw_ts, filtered_ts] = test_rows[[str(j) for j in range(1,1753)]].values.tolist()
    #     print(filtered_ts)
    #     seasonal_diff = diff(raw_ts, 1)
    #     print(seasonal_diff)
    #     double_diff = diff(seasonal_diff,1)
    #     print(double_diff)
    #     autocorrelation_plot(double_diff)
    #     plt.figure()
    #     autocorrelation_plot(seasonal_diff)
    #
    #     # plt.plot(double_diff)
    #     # plt.plot(filtered_ts)
    #     # plt.plot(seasonal_diff)
    #
    #     plt.show()

    if cow=='All':
        continue

    if counter==50:
        break

    rows = df_panting.loc[(df_panting["Cow"] == cow)]

    #x_cow, y_cow = build_prev_x_data(1, 1200, rows)
    #x_cow, y_cow = build_prev_15_data(1, 960, rows)
    x_cow, y_cow = build_combined_data(1, 960, rows)
    x.extend(x_cow)
    y.extend(y_cow)

    #x_test_cow, y_test_cow = build_prev_x_data(1200, 1620, rows)
    #x_test_cow, y_test_cow = build_prev_15_data(960, 1380, rows)
    x_test_cow, y_test_cow = build_combined_data(960, 1380, rows)
    x_test.extend(x_test_cow)
    y_test.extend(y_test_cow)

x = np.array(x)
y = np.array(y)
print(np.shape(x))
print(np.shape(y))
print(x[0])
print(y)

X = sm.add_constant(x)
results = sm.OLS(y,X).fit()

print(results.summary())

y_pred = results.predict(X)
print("In sample RMSE: " + str(mean_squared_error(y, y_pred, squared=False)))
print("In sample R-squared: " + str(r2_score(y, y_pred)))

X_test = sm.add_constant(x_test)
y_test_pred = results.predict(X_test)
print("Out of sample RMSE: " + str(mean_squared_error(y_test, y_test_pred, squared=False)))
print("Out of sample R-squared: " + str(r2_score(y_test, y_test_pred)))

coeffs = results.params
pvals = results.pvalues

fig, ax1 = plt.subplots()
ax1.plot(coeffs)
ax2 = ax1.twinx()
ax2.plot(pvals, 'ro')

fig.tight_layout()
plt.show()

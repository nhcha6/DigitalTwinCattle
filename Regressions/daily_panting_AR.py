import sys
sys.path.insert(1, '/Users/nicolaschapman/Documents/DigitalTwinCattle/DigitalTwinCattle/TrendsAndFiltering')

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tsa.api import VAR
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import adfuller
from statsmodels.tools.eval_measures import rmse, aic
from filter_data import *
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

def compute_daily_timeseries(cows, df):
    daily_timeseries_list = []
    for cow in cows:
        if cow == 'All':
            continue
        raw_series = df[(df["Cow"] == cow) & (df["Data Type"] == "panting raw")]
        raw_series = raw_series[[str(j) for j in range(1, 1753)]].values.tolist()[0]
        daily_frequencies = []
        for i in range(0, 1752, 24):
            freq = sum(raw_series[i:i+24])
            daily_frequencies.append(freq)
        daily_timeseries_list.append(daily_frequencies)
    return daily_timeseries_list

def single_cow_AR(series, lags, error_horizon, plot):
    # train and test
    train = series[0:-7]
    test = series[-7:]

    # train model
    mod = AutoReg(train, lags, old_names=False)
    res = mod.fit()
    # print(res.summary())
    forecast = res.forecast(7)
    # RMSE
    error = mean_squared_error(test[0:error_horizon], forecast[0:error_horizon], squared=False)


    if plot:
        print("Out of sample R-squared: " + str(error))
        # plot results
        plt.figure()
        plt.plot(forecast, label='forecast')
        plt.plot(test, label='actual')
        plt.legend()
        plt.show()

    # prediction = res.predict(1702,1725,dynamic=True)
    # plt.plot(prediction)

    # prediction = res.predict(1702,1725)
    # plt.plot(prediction)

    return error, forecast

def run_single_cow_AR(daily_series, lag, horizon, train_size, plot_forecast):
    all_errors = []
    counter = 0
    for series in daily_series:
        counter += 1
        series = series[0:train_size]

        error, forecast = single_cow_AR(series, lag, horizon, plot_forecast)
        all_errors.append(error)

    print("Norm RMSE: " + str(np.mean(all_errors)))
    # plt.figure()
    # plt.plot(norm_original_errors)
    # plt.show()

    return np.mean(all_errors)

def run_concat_cow_AR(daily_series, lags, error_horizon, train_size, plot_forecast):
    new_series = []
    test = []
    counter = 0
    # iterate through all cows
    for series in daily_series:
        counter += 1
        # extract all values except the last 18
        series = series[0:train_size]
        # all but last 24 (which we will predict) added to new series
        new_series.extend(series[0:-2])
        # append remaining 24 for prediction testing
        test.append(series[-2:])

    # train model
    mod = AutoReg(new_series, lags, old_names=False)
    res = mod.fit()
    # print(res.summary())

    # test each series
    count = 0
    errors = []
    orig_errors = []
    # to forecast the next 24 hour of each animal, we must loop through the concatenated data.
    for i in range(train_size-2, len(new_series), train_size-2):
        # forecast
        forecast = res.predict(i, i + 1, dynamic=True)
        # RMSE
        error = mean_squared_error(test[count][0:error_horizon], forecast[0:error_horizon], squared=False)

        # plot errors
        if plot_forecast:
            print("RMSE: " + str(error))
            # plot results
            plt.figure()
            plot_forecast = new_series[i-15:i-1]
            plot_forecast.extend(forecast)
            plot_actual = new_series[i-15:i-1]
            plot_actual.extend(test[count])
            plt.plot(plot_forecast, label='forecast')
            plt.plot(plot_actual, label='actual')
            plt.axvline(x=13, c='r')
            plt.legend()
            plt.show()

        # update variables
        count += 1
        errors.append(error)


    print("Mean RMSE: " + str(np.mean(errors)))
    plt.show()

    return np.mean(errors)

# import data
panting_df = pd.read_csv("Clean Dataset Output/panting_timeseries.csv")

# extract ordered list of animals and data type
cow_list = list(set(panting_df["Cow"]))
cow_list = sorted(cow_list)
#cow_list = ['8027107', '8022092', '8027476', '8032505', '8045911']
#cow_list = ['8027107']
data_type_list = list(set(panting_df["Data Type"]))
data_type_list = sorted(data_type_list)

panting_daily_ts = compute_daily_timeseries(cow_list, panting_df)

# run_single_cow_AR(panting_daily_ts, 21, 2, 60, False)

for lag in [2,5,10,20,30,35,40]:
    print('Lag: ' + str(lag))
    run_concat_cow_AR(panting_daily_ts, lag, 1, 61, False)


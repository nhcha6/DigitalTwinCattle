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

def diff(series, lag):
    log_series=series
    diff = [(log_series[j]-log_series[j-lag]) for j in range(lag,len(log_series))]
    return diff

def compute_daily_timeseries(cows, df, filtered=False):
    daily_timeseries_list = []
    daily_timeseries_dict = {}
    for cow in cows:
        if cow == 'All':
            continue
        raw_series = df[(df["Cow"] == cow) & (df["Data Type"] == "panting raw")]
        raw_series = raw_series[[str(j) for j in range(1, 1753)]].values.tolist()[0]
        # if filtered use filtered
        if filtered:
            raw_series = df[(df["Cow"] == cow) & (df["Data Type"] == "panting filtered")]
            raw_series = raw_series[[str(j) for j in range(1, 1753)]].values.tolist()[0]

        daily_frequencies = []
        for i in range(0, 1752, 24):
            freq = sum(raw_series[i:i+24])
            daily_frequencies.append(freq)
        daily_timeseries_list.append(daily_frequencies)
        daily_timeseries_dict[cow] = daily_frequencies
    daily_timeseries_df = pd.DataFrame.from_dict(daily_timeseries_dict, orient='index')
    return daily_timeseries_list, daily_timeseries_df

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
    # forecast = res.predict()
    # plt.plot(forecast)
    # plt.plot(new_series)
    # plt.show()

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


# run_single_cow_AR(panting_daily_ts, 21, 2, 60, False)
def error_plots(horizon):
    errors = []
    lags = []
    for train_size in [i for i in range(20, 74)]:
        min_error = 1000
        min_lag = 0

        for lag in [2, 5, 10, 20, 30, 35]:
            if lag / train_size > 0.5:
                continue
            # print("Lag: " + str(lag))
            print("Train Size: " + str(train_size))
            error = run_concat_cow_AR(panting_daily_ts, lag, horizon, train_size, False)
            if error < min_error:
                min_error = error
                min_lag = lag
        errors.append(error)
        lags.append(min_lag)

    plt.plot(errors)
    plt.figure()
    plt.plot(lags)
    plt.show()

# import data
panting_df = pd.read_csv("Clean Dataset Output/panting_timeseries.csv")

# extract ordered list of animals and data type
cow_list = list(set(panting_df["Cow"]))
cow_list = sorted(cow_list)
#cow_list = ['8027107', '8022092', '8027476', '8032505', '8045911']
#cow_list = ['8027107']
data_type_list = list(set(panting_df["Data Type"]))
data_type_list = sorted(data_type_list)

panting_daily_ts, df_panting_daily = compute_daily_timeseries(cow_list, panting_df, True)
print(df_panting_daily)
# print(len(panting_daily_ts))
# print(len(panting_daily_ts[0]))
# print(np.mean(panting_daily_ts)+np.std(panting_daily_ts))
fp_list = []
fn_list = []
predict_list = []
for i in range(9,71):
    prev_day = df_panting_daily[[i-3, i-2, i-2, i]]
    prev_day['mean'] = prev_day.mean(axis=1)
    prev_day = prev_day["mean"]
    current_day = df_panting_daily[i+1]
    # calculate false positive and false negative above and below threshold
    total_pos = 0
    total_neg = 0
    false_neg = 0
    false_pos = 0
    for index in prev_day.index:
        freq_forecast = prev_day.loc[index]
        freq_actual = current_day.loc[index]
        # print(index)
        # print(freq_forecast)
        # print(freq_actual)
        if freq_actual > 158:
            total_pos += 1
            if freq_forecast < 158:
                # plot = True
                false_neg += 1
        else:
            total_neg += 1
            if freq_forecast > 158:
                # plot = True
                false_pos += 1

    fp_list.append(false_pos/total_neg)
    fn_list.append(false_neg/total_pos)

    freq_forecast_df = prev_day.sort_values(ascending=False)
    freq_actual_df = current_day.sort_values(ascending=False)

    top_20_forecast = set(freq_forecast_df.iloc[0:20].index)
    top_20_actual = set(freq_actual_df.iloc[0:20].index)
    top_20_predicted = [x for x in top_20_forecast if x in top_20_actual]
    predict_list.append(len(top_20_predicted))

print('false neg mean: ' + str(np.mean(fn_list)))
print('false pos mean: ' + str(np.mean(fp_list)))
print('predictions mean: ' + str(np.mean(predict_list)))

plt.figure()
plt.title("top 20 predictions")
plt.plot(predict_list)
plt.figure()
plt.title("false positives")
plt.plot(fp_list)
plt.figure()
plt.title("false negatives")
plt.plot(fn_list)
plt.show()


# generate diff of simply using previous value
# mean_diffs = []
# for i in range(18,73):
#     diffs = []
#     for series in panting_daily_ts:
#         difference = series[i]-series[i-1]
#         diffs.append(difference)
#     mean_diffs.append(np.mean([abs(x) for x in diffs]))
# plt.plot(mean_diffs)
# plt.show()

#run_concat_cow_AR(panting_daily_ts, 10, 1, 73, False)
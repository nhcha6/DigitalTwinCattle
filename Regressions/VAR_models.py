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
from sklearn.metrics import mean_squared_error, r2_score

def diff(series, lag):
    log_series=series
    diff = [(log_series[j]-log_series[j-lag]) for j in range(lag,len(log_series))]
    return diff

def inverse_diff(series, init_list, lag):
    inv_diff = init_list
    for i in range(lag,len(series)+lag):
        inv_diff.append(inv_diff[i-lag] + series[i-lag])
    return inv_diff

def adfuller_test(series, signif=0.05, name='', verbose=False):
    """Perform ADFuller to test for Stationarity of given series and print report"""
    r = adfuller(series, autolag='t-stat')
    output = {'test_statistic':round(r[0], 4), 'pvalue':round(r[1], 4), 'n_lags':round(r[2], 4), 'n_obs':r[3]}
    p_value = output['pvalue']
    def adjust(val, length= 6): return str(val).ljust(length)

    # Print Summary
    print(f'    Augmented Dickey-Fuller Test on "{name}"', "\n   ", '-'*47)
    print(f' Null Hypothesis: Data has unit root. Non-Stationary.')
    print(f' Significance Level    = {signif}')
    print(f' Test Statistic        = {output["test_statistic"]}')
    print(f' No. Lags Chosen       = {output["n_lags"]}')

    for key,val in r[4].items():
        print(f' Critical value {adjust(key)} = {round(val, 3)}')

    if p_value <= signif:
        print(f" => P-Value = {p_value}. Rejecting Null Hypothesis.")
        print(f" => Series is Stationary.")
    else:
        print(f" => P-Value = {p_value}. Weak evidence to reject the Null Hypothesis.")
        print(f" => Series is Non-Stationary.")

def stationarity_tests(df_panting):
    filtered_panting = df_panting[(df_panting["Cow"] == '8021870') & (df_panting["Data Type"] == "panting filtered")]
    filtered_panting = filtered_panting[[str(j) for j in range(1, 1753)]].values.tolist()[0]

    adfuller_test(filtered_panting)

    diff_1 = diff(filtered_panting, 24)
    adfuller_test(diff_1)

def filtered_data_generation(df_panting):
    raw_panting = df_panting[(df_panting["Cow"] == '8021870') & (df_panting["Data Type"] == "panting filtered")]
    raw_panting = raw_panting[[str(j) for j in range(1, 1753)]].values.tolist()[0]
    filtered_panting = df_panting[(df_panting["Cow"] == '8021870') & (df_panting["Data Type"] == "panting filtered")]
    filtered_panting = filtered_panting[[str(j) for j in range(1, 1753)]].values.tolist()[0]

    new_filtered = []
    for i in range(0, len(raw_panting) - 24):
        seq = raw_panting[i:i + 24]
        filt_seq = butter_lp_filter([3], [4], seq, "All", False)
        # print(seq)
        # print(filt_seq)

        if i == 0:
            new_filtered.extend(filt_seq)
        else:
            new_filtered.append(filt_seq[-1])

        # plt.plot(new_filtered)
        # plt.plot(filtered_panting[0:i + 24])
        # plt.show()

    plt.plot(new_filtered)
    plt.plot(filtered_panting)
    plt.show()

def single_cow_AR(series, lags, error_horizon):
    # train and test
    train = series[0:-24]
    test = series[-24:]

    # train model
    mod = AutoReg(train, lags, old_names=False)
    res = mod.fit()
    # print(res.summary())
    forecast = res.forecast(24)
    # RMSE
    error = mean_squared_error(test[0:error_horizon], forecast[0:error_horizon])

    # if error>0.5:
    #     print("Out of sample R-squared: " + str(error))
    #     # plot results
    #     plt.figure()
    #     plt.plot(forecast, label='forecast')
    #     plt.plot(test, label='actual')
    #     plt.legend()

    # prediction = res.predict(1702,1725,dynamic=True)
    # plt.plot(prediction)

    # prediction = res.predict(1702,1725)
    # plt.plot(prediction)

    return error, forecast

def run_concat_cow_AR(df_panting, cows, lags, error_horizon):
    new_series = []
    test = []
    predict_ts = []
    init_i = []
    init_ii = []
    counter = 0
    # iterate through all cows
    for cow in cows:
        counter += 1
        if cow == "All":
            continue
        # extract all values except the last 18
        filtered_panting = df_panting[(df_panting["Cow"] == cow) & (df_panting["Data Type"] == "panting filtered")]
        filtered_panting = filtered_panting[[str(j) for j in range(1, 1735)]].values.tolist()[0]
        differenced = diff(filtered_panting, 1)
        double_diff = diff(differenced, 1)
        # all but last 24 (which we will predict) added to new series
        new_series.extend(double_diff[0:-24])
        # append remaining 24 for prediction testing
        test.append(double_diff[-24:])
        # append intial difference and filtered values for rebuilding of data
        init_i.append(differenced[-25])
        init_ii.append(filtered_panting[-26])
        # append all original data to be predicted and preceding 6 hours
        predict_ts.append(filtered_panting[-30:])

    # train model
    mod = AutoReg(new_series, lags, old_names=False)
    res = mod.fit()
    # print(res.summary())

    # test each series
    count = 0
    errors = []
    orig_errors = []
    # to forecast the next 24 hour of each animal, we must loop throught the concatenated data.
    for i in range(1708, 338184, 1708):
        # forecast
        forecast = res.predict(i, i + 23, dynamic=True)
        # RMSE
        error = mean_squared_error(test[count][0:error_horizon], forecast[0:error_horizon])

        # plot errors
        if False:
            print("Out of sample R-squared: " + str(error))
            # plot results
            plt.figure()
            plt.plot(forecast, label='forecast')
            plt.plot(test[count], label='actual')
            plt.legend()

        # recreate original sequence prediction and compare to the actual data
        forecast_i = inverse_diff(forecast, [init_i[count]], 1)
        forecast_ii = inverse_diff(forecast_i, [init_ii[count]], 1)

        orig_error = mean_squared_error(forecast_ii[-24:-24+error_horizon], predict_ts[count][-24:-24+error_horizon])

        # plot forecast of original data
        if False:
            plt.figure()
            plt.plot(predict_ts[count][0:6] + forecast_ii[-24:], label='forecast')
            plt.plot(predict_ts[count], label='actual')
            plt.axvline(x=5, c='r')
            plt.legend()
            plt.show()

        # update variables
        count += 1
        errors.append(error)
        orig_errors.append(orig_error)

    print("Mean RMSE: " + str(np.mean(orig_errors)))
    plt.show()

def run_single_cow_AR(cows, df_panting, lag, horizon):
    all_errors = []
    original_errors = []
    counter = 0
    for cow in cows:
        counter += 1

        filtered_panting = df_panting[(df_panting["Cow"] == cow) & (df_panting["Data Type"] == "panting filtered")]
        filtered_panting = filtered_panting[[str(j) for j in range(1, 1735)]].values.tolist()[0]
        differenced = diff(filtered_panting, 1)
        double_diff = diff(differenced, 1)

        error, forecast = single_cow_AR(double_diff, lag, horizon)
        all_errors.append(error)

        #recreate original sequence prediction and compare to the actual data
        init_i = [differenced[-25]]
        init_ii = [filtered_panting[-26]]
        forecast_i = inverse_diff(forecast, init_i,1)
        forecast_ii = inverse_diff(forecast_i, init_ii, 1)

        original_errors.append(mean_squared_error(filtered_panting[-24:-24+horizon], forecast_ii[-24:-24+horizon]))

        # plt.plot(filtered_panting[-30:-24] + forecast_ii[-24:],label='forecast')
        # plt.plot(filtered_panting[-30:], label='actual')
        # plt.axvline(x=5, c='r')
        # plt.legend()
        # plt.show()

    print("Mean RMSE: " + str(np.mean(original_errors)))
    plt.show()

def indivual_AR_original_signal(df_panting, cow_list, lag, horizon):
    errors = []
    for cow in cow_list:
        filtered_panting = df_panting[(df_panting["Cow"] == cow) & (df_panting["Data Type"] == "panting filtered")]
        filtered_panting = filtered_panting[[str(j) for j in range(1, 1735)]].values.tolist()[0]
        error, forecast = single_cow_AR(filtered_panting, lag, horizon)
        # print('RMSE: ' + str(error))
        errors.append(error)
        x = filtered_panting[-30:-24]
        x.extend(forecast)
        plt.plot(x,label='forecast')
        plt.plot(filtered_panting[-30:], label='actual')
        plt.axvline(x=5, c='r')
        plt.legend()
        plt.show()

    print("Mean RMSE: " + str(np.mean(errors)))

# import data
panting_df = pd.read_csv("Clean Dataset Output/panting_timeseries.csv")

# extract ordered list of animals and data type
cow_list = list(set(panting_df["Cow"]))
cow_list = sorted(cow_list)
#cow_list = ['8027107', '8022092', '8027476', '8032505', '8045911']
#cow_list = ['8027107']
data_type_list = list(set(panting_df["Data Type"]))
data_type_list = sorted(data_type_list)

# stationarity tests
# stationarity_tests(df_panting)

# filtering tests
# filtered_data_generation(df_panting)

# basic autoregression of filtered data
#run_single_cow_AR(cow_list, panting_df, 108, 6)

# concatenate timeseries
#run_concat_cow_AR(panting_df, cow_list, 36, 12)

# not using difference in difference
# indivual_AR_original_signal(panting_df, cow_list, 36, 12)

# for testing multiple lags and horizons
for lag in [24, 96, 108,120,132,144]:
    for horizon in [6]:
        print("lag: " + str(lag))
        run_single_cow_AR(cow_list, panting_df, lag, horizon)


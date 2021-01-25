import sys
sys.path.insert(1, '/Users/nicolaschapman/Documents/DigitalTwinCattle/DigitalTwinCattle/TrendsAndFiltering')

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

warnings.filterwarnings("ignore")

cutoff_dict = {'panting raw': 3, 'resting raw': 4.5, 'medium activity raw': 4}


def invert_transformation(df_train, df_forecast, second_diff=False):
    """Revert back the differencing to get the forecast to original scale."""
    df_fc = df_forecast.copy()
    columns = df_train.columns
    for col in columns:
        # Roll back 2nd Diff
        if second_diff:
            df_fc[str(col)+'_1d'] = (df_train[col].iloc[-1]-df_train[col].iloc[-2]) + df_fc[str(col)+'_2d'].cumsum()
        # Roll back 1st Diff
        df_fc[str(col)+'_forecast'] = df_train[col].iloc[-1] + df_fc[str(col)+'_1d'].cumsum()
    return df_fc

def diff(series, lag):
    log_series=series
    diff = [(log_series[j]-log_series[j-lag]) for j in range(lag,len(log_series))]
    return diff

def inverse_diff(series, init_list, lag):
    inv_diff = init_list
    for i in range(lag,len(series)+lag):
        inv_diff.append(inv_diff[i-lag] + series[i-lag])
    return inv_diff

def grangers_causation_matrix(data, variables, test='ssr_chi2test', verbose=False):
    """Check Granger Causality of all possible combinations of the Time series.
    The rows are the response variable, columns are predictors. The values in the table
    are the P-Values. P-Values lesser than the significance level (0.05), implies
    the Null Hypothesis that the coefficients of the corresponding past values is
    zero, that is, the X does not cause Y can be rejected.

    data      : pandas dataframe containing the time series variables
    variables : list containing names of the time series variables.
    """
    maxlag = 12
    df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    for c in df.columns:
        for r in df.index:
            test_result = grangercausalitytests(data[[r, c]], maxlag=maxlag, verbose=False)
            p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
            if verbose: print(f'Y = {r}, X = {c}, P Values = {p_values}')
            min_p_value = np.min(p_values)
            df.loc[r, c] = min_p_value
    df.columns = [var + '_x' for var in variables]
    df.index = [var + '_y' for var in variables]
    return df

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

def run_single_cow_AR(cows, df_panting, lag, horizon, train_size, plot_forecast):
    all_errors = []
    original_errors = []
    counter = 0
    for cow in cows:
        counter += 1

        if cow == "All":
            continue

        filtered_panting = df_panting[(df_panting["Cow"] == cow) & (df_panting["Data Type"] == "panting filtered")]
        filtered_panting = filtered_panting[[str(j) for j in range(1, train_size)]].values.tolist()[0]
        differenced = diff(filtered_panting, 1)
        double_diff = diff(differenced, 1)

        error, forecast = single_cow_AR(double_diff, lag, horizon)
        all_errors.append(error)

        #recreate original sequence prediction and compare to the actual data
        init_i = [differenced[-25]]
        init_ii = [filtered_panting[-26]]
        forecast_i = inverse_diff(forecast, init_i,1)
        forecast_ii = inverse_diff(forecast_i, init_ii, 1)

        original_error = mean_squared_error(filtered_panting[-24:-24+horizon], forecast_ii[-24:-24+horizon], squared=False)
        max_val = max(filtered_panting[-24:-24+horizon])
        #norm_orig_error = mean_absolute_percentage_error(filtered_panting[-24:-24+horizon], forecast_ii[-24:-24+horizon])
        norm_orig_error = original_error/max_val
        # print(original_error)
        if max_val>1:
            original_errors.append(norm_orig_error)

        if cow == 'All':
            print(norm_orig_error)
            plt.plot(filtered_panting[-30:-24] + forecast_ii[-24:],label='forecast')
            plt.plot(filtered_panting[-30:], label='actual')
            plt.axvline(x=5, c='r')
            plt.legend()
            plt.show()

    print("Norm RMSE: " + str(np.mean(original_errors)))
    # plt.figure()
    # plt.plot(norm_original_errors)
    # plt.show()

    return np.mean(original_errors)

def run_VAR(df_list, model_data, cows, train_size, lag, horizon, plot=False, df_weather = None):
    combined_df = pd.concat([df for df in df_list])
    errors = []
    prev_cow = '8048118'
    for cow in cows:
        if cow == 'All':
            continue

        cow_df = combined_df[combined_df["Cow"] == cow]
        VAR_df = cow_df.loc[cow_df['Data Type'].isin(model_data)]

        # include herd data
        if 'herd' in model_data:
            all_df = combined_df[combined_df["Cow"] == "All"]
            all_df = all_df.loc[all_df['Data Type'] == 'panting filtered']
            all_df["Data Type"] = 'herd'
            VAR_df = pd.concat([VAR_df, all_df])
        if 'prev' in model_data:
            all_df = combined_df[combined_df["Cow"] == prev_cow]
            all_df = all_df.loc[all_df['Data Type'] == 'panting filtered']
            all_df["Data Type"] = 'prev'
            VAR_df = pd.concat([VAR_df, all_df])
            prev_cow = cow

        # diff series
        new_col = VAR_df["Data Type"]
        VAR_df = VAR_df.transpose()
        VAR_df.columns = new_col
        VAR_df = VAR_df.iloc[2:,:]

        # add weather data if selected
        if 'HLI' in model_data:
            VAR_df = VAR_df.join(df_weather.set_index(VAR_df.index))
            VAR_df = VAR_df.drop(['Time', 'Date', 'THI'],axis=1)
        # add weather data if selected
        if 'THI' in model_data:
            VAR_df = VAR_df.join(df_weather.set_index(VAR_df.index))
            VAR_df = VAR_df.drop(['Time', 'Date', 'HLI'], axis=1)

        VAR_d_df = VAR_df.diff().dropna()
        VAR_dd_df = VAR_d_df.diff().dropna()
        # split to test and train
        train_diff_df = VAR_dd_df.iloc[0:train_size-2,:].astype(float)
        test_diff_df = VAR_dd_df.iloc[train_size-2:train_size+22,:].astype(float)
        train_df = VAR_df.iloc[0:train_size,:]
        test_df = VAR_df.iloc[train_size:train_size+24,:]

        # run causality test
        if False:
            grangers_df = grangers_causation_matrix(VAR_dd_df, model_data)
            print(grangers_df.to_string())

        # create model
        model = VAR(train_diff_df)
        result = model.fit(lag)

        # forecast result
        forecast_input = train_diff_df.values[-lag:]
        forecast = result.forecast(y=forecast_input, steps=24)

        # rebuild input
        df_forecast = pd.DataFrame(forecast, index=test_diff_df.index, columns=test_diff_df.columns+'_2d')
        df_forecast = invert_transformation(train_df, df_forecast, True)
        plot_var = model_data[0]

        # calculate RMSE
        error = mean_squared_error(test_df[plot_var].iloc[0:horizon], df_forecast[plot_var + "_forecast"].iloc[0:horizon], squared=False)
        max_val = max(test_df[plot_var])
        # norm_orig_error = mean_absolute_percentage_error(filtered_panting[-24:-24+horizon], forecast_ii[-24:-24+horizon])
        norm_error = error / max_val
        # print(original_error)
        if max_val > 1:
            errors.append(norm_error)

        # if True plot
        if plot:
            # plt.figure()
            # plt.plot(df_forecast[plot_var + "_2d"], label='forecast')
            # plt.plot(test_diff_df[plot_var], label='actual')
            # plt.legend()
            print(norm_error)
            for header in model_data:
                plt.figure()
                plt.title(header)
                plt.plot(df_forecast[header + "_forecast"], label='forecast')
                plt.plot(test_df[header], label='actual')
                plt.legend()
                plt.show()

    # return the mean error
    mean_error = np.mean(errors)
    print("Mean RMSE: " + str(mean_error))
    return mean_error

def run_VAR_averaged(df_list, models, cows, train_size, lag, horizon, plot=False, df_weather=None, filter_lag=None):
    combined_df = pd.concat([df for df in df_list])
    # print(combined_df[combined_df['Cow']=='8048118'].iloc[:,train_size-10:train_size + 2])
    # if filter lag input, we need to adjust the panting data
    if filter_lag is not None:
        for data_type in set([item for sublist in models for item in sublist]):
            for cow in cows:
                if 'filtered' in data_type:
                    raw_data_type = data_type.replace("filtered", "raw")
                    # print(combined_df[(combined_df["Cow"] == cow) & (combined_df["Data Type"] == data_type)].iloc[:,train_size + 2 - 10:train_size + 2])
                    raw_data = combined_df[(combined_df["Cow"] == cow) & (combined_df["Data Type"] == raw_data_type)].iloc[:,2:train_size + 2 + filter_lag].values[0]
                    # print(raw_data)
                    filt_seq = butter_lp_filter([cutoff_dict[raw_data_type]], [4], raw_data, "All", False)
                    # print(filt_seq[-10:-filter_lag])
                    combined_df.loc[(combined_df["Cow"] == cow) & (combined_df["Data Type"] == data_type),2:train_size + 2] = filt_seq[0:-filter_lag]
                    # print(combined_df[(combined_df["Cow"] == cow) & (combined_df["Data Type"] == data_type)].iloc[:,train_size + 2 - 10:train_size + 2])

    # print(combined_df[combined_df['Cow'] == '8048118'].iloc[:, train_size-10:train_size + 2])

    errors = []
    prev_cow = '8048118'
    for cow in cows:
        if cow == 'All':
            continue

        cow_df = combined_df[combined_df["Cow"] == cow]

        forecast_list = []
        for model_data in models:

            VAR_df = cow_df.loc[cow_df['Data Type'].isin(model_data)]

            # include herd data
            if 'herd' in model_data:
                all_df = combined_df[combined_df["Cow"] == "All"]
                all_df = all_df.loc[all_df['Data Type']=='panting filtered']
                all_df["Data Type"] = all_df["Data Type"] + ' herd'
                VAR_df = pd.concat([VAR_df, all_df])
            if 'prev' in model_data:
                all_df = combined_df[combined_df["Cow"] == prev_cow]
                all_df = all_df.loc[all_df['Data Type'] == 'panting filtered']
                all_df["Data Type"] = all_df["Data Type"] + ' herd'
                VAR_df = pd.concat([VAR_df, all_df])
                prev_cow = cow

            # diff series
            new_col = VAR_df["Data Type"]
            VAR_df = VAR_df.transpose()
            VAR_df.columns = new_col
            VAR_df = VAR_df.iloc[2:,:]

            # add weather data if selected
            if 'HLI' in model_data:
                VAR_df = VAR_df.join(df_weather.set_index(VAR_df.index))
                VAR_df = VAR_df.drop(['Time', 'Date', 'THI'], axis=1)
            # add weather data if selected
            if 'THI' in model_data:
                VAR_df = VAR_df.join(df_weather.set_index(VAR_df.index))
                VAR_df = VAR_df.drop(['Time', 'Date', 'HLI'], axis=1)

            VAR_d_df = VAR_df.diff().dropna()
            VAR_dd_df = VAR_d_df.diff().dropna()
            # split to test and train
            train_diff_df = VAR_dd_df.iloc[0:train_size-2,:].astype(float)
            test_diff_df = VAR_dd_df.iloc[train_size-2:train_size+22,:].astype(float)
            train_df = VAR_df.iloc[0:train_size,:]
            test_df = VAR_df.iloc[train_size:train_size+24,:]

            # run causality test
            if False:
                grangers_df = grangers_causation_matrix(VAR_dd_df, model_data)
                print(grangers_df.to_string())

            # create model
            model = VAR(train_diff_df)
            result = model.fit(lag)

            # forecast result
            forecast_input = train_diff_df.values[-lag:]
            forecast = result.forecast(y=forecast_input, steps=24)

            # rebuild input
            df_forecast = pd.DataFrame(forecast, index=test_diff_df.index, columns=test_diff_df.columns+'_2d')
            df_forecast = invert_transformation(train_df, df_forecast, True)
            plot_var = model_data[0]

            forecast_list.append(df_forecast[plot_var + "_forecast"].values)

        # average all lists
        average_forecast = [np.mean(x) for x in zip(*forecast_list)]

        # calculate RMSE
        error = mean_squared_error(test_df[plot_var].iloc[filter_lag:horizon+filter_lag], average_forecast[filter_lag:horizon+filter_lag], squared=False)
        max_val = max(test_df[plot_var])
        # norm_orig_error = mean_absolute_percentage_error(filtered_panting[-24:-24+horizon], forecast_ii[-24:-24+horizon])
        norm_error = error / max_val
        # print(original_error)
        if max_val > 1:
            errors.append(norm_error)

        # if True plot
        if plot:
            # plt.figure()
            # plt.plot(df_forecast[plot_var + "_2d"], label='forecast')
            # plt.plot(test_diff_df[plot_var], label='actual')
            # plt.legend()
            print(norm_error)
            plt.figure()
            plt.title(plot_var)
            plt.plot(average_forecast, label='forecast')
            plt.plot(test_df[plot_var], label='actual')
            plt.legend()
            plt.show()

    # return the mean error
    mean_error = np.mean(errors)
    print("Mean RMSE: " + str(mean_error))
    return mean_error

def run_VAR_averaged_frequency(df_list, models, cows, train_size, estimate_time, lag, horizon, plot=False, df_weather=None, filter_lag = None):
    combined_df = pd.concat([df for df in df_list])
    errors = []
    prev_cow = '8048118'

    # if filter lag input, we need to adjust the panting data
    if filter_lag is not None:
        for data_type in set([item for sublist in models for item in sublist]):
            for cow in cows:
                if 'filtered' in data_type:
                    raw_data_type = data_type.replace("filtered", "raw")
                    # print(combined_df[(combined_df["Cow"] == cow) & (combined_df["Data Type"] == data_type)].iloc[:,train_size + 2 - 10:train_size + 2])
                    raw_data = \
                    combined_df[(combined_df["Cow"] == cow) & (combined_df["Data Type"] == raw_data_type)].iloc[:,
                    2:train_size + 2 + filter_lag].values[0]
                    # print(raw_data)
                    filt_seq = butter_lp_filter([cutoff_dict[raw_data_type]], [4], raw_data, "All", False)
                    # print(filt_seq[-10:-filter_lag])
                    combined_df.loc[(combined_df["Cow"] == cow) & (combined_df["Data Type"] == data_type),
                    2:train_size + 2] = filt_seq[0:-filter_lag]
                    # print(combined_df[(combined_df["Cow"] == cow) & (combined_df["Data Type"] == data_type)].iloc[:,train_size + 2 - 10:train_size + 2])

    # error calc
    false_pos = 0
    false_neg = 0
    total_pos = 0
    total_neg = 0

    # frequency rank
    freq_forecast_dict = {}
    freq_actual_dict = {}

    for cow in cows:
        if cow == 'All':
            continue

        cow_df = combined_df[combined_df["Cow"] == cow]

        forecast_list = []
        for model_data in models:

            VAR_df = cow_df.loc[cow_df['Data Type'].isin(model_data)]

            # include herd data
            if 'herd' in model_data:
                all_df = combined_df[combined_df["Cow"] == "All"]
                all_df = all_df.loc[all_df['Data Type']=='panting filtered']
                all_df["Data Type"] = all_df["Data Type"] + ' herd'
                VAR_df = pd.concat([VAR_df, all_df])
            if 'prev' in model_data:
                all_df = combined_df[combined_df["Cow"] == prev_cow]
                all_df = all_df.loc[all_df['Data Type'] == 'panting filtered']
                all_df["Data Type"] = all_df["Data Type"] + ' herd'
                VAR_df = pd.concat([VAR_df, all_df])
                prev_cow = cow

            # diff series
            new_col = VAR_df["Data Type"]
            VAR_df = VAR_df.transpose()
            VAR_df.columns = new_col
            VAR_df = VAR_df.iloc[2:,:]

            # add weather data if selected
            if 'HLI' in model_data:
                VAR_df = VAR_df.join(df_weather.set_index(VAR_df.index))
                VAR_df = VAR_df.drop(['Time', 'Date', 'THI'], axis=1)
            # add weather data if selected
            if 'THI' in model_data:
                VAR_df = VAR_df.join(df_weather.set_index(VAR_df.index))
                VAR_df = VAR_df.drop(['Time', 'Date', 'HLI'], axis=1)

            VAR_d_df = VAR_df.diff().dropna()
            VAR_dd_df = VAR_d_df.diff().dropna()
            # split to test and train
            train_diff_df = VAR_dd_df.iloc[0:train_size-2,:].astype(float)
            test_diff_df = VAR_dd_df.iloc[train_size-2:train_size+22,:].astype(float)
            train_df = VAR_df.iloc[0:train_size,:]
            test_df = VAR_df.iloc[train_size:train_size+24,:]

            # run causality test
            if False:
                grangers_df = grangers_causation_matrix(VAR_dd_df, model_data)
                print(grangers_df.to_string())

            # create model
            model = VAR(train_diff_df)
            result = model.fit(lag)

            # forecast result
            forecast_input = train_diff_df.values[-lag:]
            forecast = result.forecast(y=forecast_input, steps=24)

            # rebuild input
            df_forecast = pd.DataFrame(forecast, index=test_diff_df.index, columns=test_diff_df.columns+'_2d')
            df_forecast = invert_transformation(train_df, df_forecast, True)
            plot_var = model_data[0]

            forecast_list.append(df_forecast[plot_var + "_forecast"].values)

        # average all lists
        average_forecast = [np.mean(x) for x in zip(*forecast_list)]
        freq_forecast = sum(average_forecast[0:24-estimate_time]) + sum(train_df[plot_var].iloc[-estimate_time:])
        freq_actual = sum(test_df[plot_var].iloc[0:24-estimate_time]) + sum(train_df[plot_var].iloc[-estimate_time:])

        freq_forecast_dict[cow] = freq_forecast
        freq_actual_dict[cow] = freq_actual

        # calculate false positive and false negative above and below threshold
        if freq_actual > 158:
            total_pos += 1
            if freq_forecast < 158:
                # plot = True
                false_neg +=1
        else:
            total_neg += 1
            if freq_forecast > 158:
                # plot = True
                false_pos += 1

        # if True plot
        if plot:
            # plt.figure()
            # plt.plot(df_forecast[plot_var + "_2d"], label='forecast')
            # plt.plot(test_diff_df[plot_var], label='actual')
            # plt.legend()
            print(freq_actual)
            print(freq_forecast)
            plt.figure()
            plt.title(plot_var)
            plt.plot(average_forecast, label='forecast')
            plt.plot(test_df[plot_var], label='actual')
            plt.legend()
            plt.show()
            plot = False

    freq_forecast_df = pd.DataFrame.from_dict(freq_forecast_dict, orient='index').sort_values(by=[0], ascending=False)
    freq_actual_df = pd.DataFrame.from_dict(freq_actual_dict, orient='index').sort_values(by=[0], ascending=False)
    top_20_forecast = set(freq_forecast_df.iloc[0:20,0].index)
    top_20_actual = set(freq_actual_df.iloc[0:20,0].index)
    top_20_predicted = [x for x in top_20_forecast if x in top_20_actual]

    # print(freq_forecast_df.head(20))
    # print(freq_actual_df.head(20))
    print("predicted:" + str(len(top_20_predicted)))
    print("total pos: "+ str(total_pos))
    print("false pos: " + str(false_pos/total_neg))
    print("false neg: " + str(false_neg/total_pos))
    return len(top_20_predicted), false_pos/total_neg, false_neg/total_pos, total_pos

# for testing multiple lags and horizons
def error_plot(horizon):
    error_list = []
    lag_list = []

    # spread over entire series
    #iter_list = [i for i in range(246, 1711, 24)]
    # spread over entire day
    iter_list = [i for i in range(655, 655+72, 1)]

    for train_size in iter_list:
        print("train size: " + str(train_size))
        min_error = 1000
        min_lag = 0
        for lag in [72]:
            print("lag " + str(lag))
            # calculate error for given model
            error = run_VAR(df_list, headers, cow_list, train_size, lag, horizon)

            if error < min_error:
                min_error = error
                min_lag = lag
        error_list.append(min_error)
        lag_list.append(min_lag)

    print(np.mean(error_list))

    plt.plot(error_list)
    plt.figure()
    plt.plot(lag_list)
    plt.show()

# for testing multiple models
def error_random_test(horizon, header_list, df_panting, cows):
    sizes = []
    train_size = 246
    train_size += random.randint(1, 600)
    while train_size < 1710:
        sizes.append(train_size)
        train_size += random.randint(1, 600)

    all_errors = []
    rows = []
    for data in header_list:
        print(data)
        error_list = []

        for train_size in sizes:
            print("train size: " + str(train_size))
            lag = math.ceil(train_size/500)*24
            print("lag " + str(lag))

            # calculate error for given model
            error = run_VAR(df_list, data, cow_list, train_size, lag, horizon, False, weather_df)
            error_list.append(error)

        error_list.append(np.mean(error_list))
        all_errors.append(error_list)
        rows.append(data[-1])
        print('Mean of mean RMSE: ' + str(np.mean(error_list)))

    # run regression baseline
    print("panting filtered autoregression")
    error_list = []

    for train_size in sizes:
        print("train size: " + str(train_size))
        lag = math.ceil(train_size / 500) * 24
        print("lag " + str(lag))

        # calculate error for given model
        error = run_single_cow_AR(cows, df_panting, lag, horizon, train_size+25, False)
        error_list.append(error)

    error_list.append(np.mean(error_list))
    all_errors.append(error_list)
    rows.append("autoregression")
    print('Mean of mean RMSE: ' + str(np.mean(error_list)))

    sizes.append("Mean")
    errors_df = pd.DataFrame(all_errors, columns = sizes, index = rows)

    return errors_df

# for testing multiple models when average data is to be included
def average_error_random_test(horizon, header_list):
    sizes = []
    train_size = 246
    train_size += random.randint(1, 600)
    while train_size < 1710:
        sizes.append(train_size)
        train_size += random.randint(1, 600)

    all_errors = []
    rows = []
    for data in header_list:
        print('\n'+str(data))
        error_list = []

        for train_size in sizes:
            print("train size: " + str(train_size))
            lag = math.ceil(train_size/500)*24
            print("lag " + str(lag))

            # calculate error for given model
            error = run_VAR(df_list, data, cow_list, train_size, lag, horizon, False, weather_df)
            error_list.append(error)

        error_list.append(np.mean(error_list))
        all_errors.append(error_list)
        rows.append(data[-1])
        print('Mean of mean RMSE: ' + str(np.mean(error_list)))

    # run average
    error_list = []
    print('\n Averaged')
    for train_size in sizes:
        print("train size: " + str(train_size))
        lag = math.ceil(train_size/500)*24
        print("lag " + str(lag))

        # calculate error for given model
        error = run_VAR_averaged(df_list, header_list, cow_list, train_size, lag, horizon, False, weather_df)
        error_list.append(error)

    error_list.append(np.mean(error_list))
    all_errors.append(error_list)
    rows.append(data[-1]+' + herd')
    print('Mean of mean RMSE: ' + str(np.mean(error_list)))

    sizes.append("Mean")
    errors_df = pd.DataFrame(all_errors, columns = sizes, index = rows)

    return errors_df

# for testing multiple models when average data is to be included
def average_error_compare(horizon, header_list1, header_list2):
    sizes = []
    train_size = 246
    train_size += random.randint(1, 600)
    while train_size < 1710:
        sizes.append(train_size)
        train_size += random.randint(1, 600)

    all_errors = []
    rows = []

    # run average
    error_list = []
    print('\nModel 1')
    for train_size in sizes:
        print("train size: " + str(train_size))
        lag = math.ceil(train_size/500)*24
        print("lag " + str(lag))

        # calculate error for given model
        error = run_VAR_averaged(df_list, header_list1, cow_list, train_size, lag, horizon, False, weather_df)
        error_list.append(error)

    error_list.append(np.mean(error_list))
    all_errors.append(error_list)
    rows.append('model 1')
    print('Mean of mean RMSE: ' + str(np.mean(error_list)))

    # run average
    error_list = []
    print('\nModel 2')
    for train_size in sizes:
        print("train size: " + str(train_size))
        lag = math.ceil(train_size / 500) * 24
        print("lag " + str(lag))

        # calculate error for given model
        error = run_VAR_averaged(df_list, header_list2, cow_list, train_size, lag, horizon, False, weather_df)
        error_list.append(error)

    error_list.append(np.mean(error_list))
    all_errors.append(error_list)
    rows.append('\nModel 2')
    print('Mean of mean RMSE: ' + str(np.mean(error_list)))

    sizes.append("Mean")
    errors_df = pd.DataFrame(all_errors, columns = sizes, index = rows)

    return errors_df

def lag_error_compare(horizon, header_list):
    sizes = []
    train_size = 246
    train_size += random.randint(1, 700)
    while train_size < 1710:
        sizes.append(train_size)
        train_size += random.randint(1, 700)

    all_errors = []
    rows = []

    for i in range(4,10,1):
        # run average
        error_list = []
        model_name = 'filter lag = ' + str(i)
        print('\n' + str(model_name))
        for train_size in sizes:
            print("train size: " + str(train_size))
            lag = math.ceil(train_size / 500) * 24
            print("lag " + str(lag))

            # calculate error for given model
            if i == 0:
                filt_lag = None
            else:
                filt_lag = i

            error = run_VAR_averaged(df_list, header_list, cow_list, train_size-filt_lag, lag, horizon, False, weather_df, filt_lag)
            error_list.append(error)

        error_list.append(np.mean(error_list))
        all_errors.append(error_list)
        rows.append(model_name)
        print('Mean of mean RMSE: ' + str(np.mean(error_list)))

    sizes.append("Mean")
    errors_df = pd.DataFrame(all_errors, columns=sizes, index=rows)

    return errors_df


# for predicting the top 20 heat prone animals
def predict_top_20(start, models, forecast_time, filt_lag):
    predictions = []
    fp_list = []
    fn_list = []
    pos_list = []
    # spread over entire series (start=246 is for 6am prediction)
    iter_list = [i for i in range(start, start+1465, 24)]
    # iter_list = [i for i in range(start, start+241, 24)]

    for train_size in iter_list:
        print("train size: " + str(train_size))
        lag = math.ceil(train_size / 500) * 24
        print("lag " + str(lag))

        # run_VAR_averaged(df_list, model_list, cow_list, 1710, 96, 12, False, weather_df)
        predicted, fp, fn, total_pos = run_VAR_averaged_frequency(df_list, models, cow_list, train_size-filt_lag, forecast_time-filt_lag, 96, 12, False, weather_df, filt_lag)
        predictions.append(predicted)
        fp_list.append(fp)
        fn_list.append(fn)
        pos_list.append(total_pos)

    print('false neg mean: ' + str(np.mean(fn_list)))
    print('false pos mean: ' + str(np.mean(fp_list)))
    print('predictions mean: ' + str(np.mean(predictions)))

    plt.figure()
    plt.title("top 20 predictions")
    plt.plot(predictions)
    plt.figure()
    plt.title("false positives")
    plt.plot(fp_list)
    plt.figure()
    plt.title("false negatives")
    plt.plot(fn_list)
    # plt.figure()
    # plt.title("total above 1 std from mean")
    # plt.plot(pos_list)
    plt.show()


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

# declare header and run VAR
df_list = [panting_df, resting_df, eating_df, rumination_df, medium_activity_df]

# headers = ["panting filtered", "medium activity filtered"]
headers = ["panting filtered", "THI"]
# run_VAR(df_list, headers, cow_list, 1710, 96, 12, True, weather_df)

# run_single_cow_AR(cow_list, panting_df, 96, 12, 1735, False)

model_list = [["panting filtered", "HLI"], ["panting filtered", "herd"], ["panting filtered", "medium activity filtered"], ["panting filtered", "resting filtered"]]
# model_list = [["panting filtered", "medium activity filtered"],["panting filtered", "THI"]]
# run_VAR_averaged(df_list, model_list, cow_list, 1710, 96, 6, False, weather_df, 2)
# run_VAR_averaged(df_list, model_list, cow_list, 1710, 96, 10, False, weather_df, 6)
# run_VAR_averaged(df_list, model_list, cow_list, 1710, 96, 12, False, weather_df, 8)
# run_VAR_averaged_frequency(df_list, model_list, cow_list, 248, 8, 96, 12, False, weather_df, 2)
predict_top_20(248, model_list, 8, 6)

# tests
# header_list1 = [["panting filtered", "HLI"], ["panting filtered", "THI"], ["panting filtered", "herd"], ["panting filtered", "medium activity filtered"], ["panting filtered", "resting filtered"], ["panting filtered", "eating filtered"], ["panting filtered", "prev"]]
# header_list2 = [["panting filtered", "HLI"], ["panting filtered", "herd"], ["panting filtered", "medium activity filtered"], ["panting filtered", "resting filtered"]]
# summary_list = []
# for i in range(4):
#     # df_errors = error_random_test(6, header_list, panting_df, cow_list)
#     # df_errors = average_error_random_test(12, header_list)
#     # df_errors = average_error_compare(12, header_list1, header_list2)
#     df_errors = lag_error_compare(12, model_list)
#     summary_list.append(df_errors)
# for data in summary_list:
#     print('\n')
#     print(data.to_string())
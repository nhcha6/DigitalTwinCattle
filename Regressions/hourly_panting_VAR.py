import sys
sys.path.insert(1, '/Users/nicolaschapman/Documents/DigitalTwinCattle/DigitalTwinCattle/TrendsAndFiltering')

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
from statsmodels.tools.eval_measures import rmse, aic
from filter_data import *
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from statsmodels.tsa.stattools import grangercausalitytests

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

def run_VAR(df_list, model_data, cows, train_size, lag):
    combined_df = pd.concat([df for df in df_list])
    for cow in cows:
        # create data df
        cow_df = combined_df[combined_df["Cow"]==cow]
        VAR_df = cow_df.loc[cow_df['Data Type'].isin(model_data)]
        # diff series
        new_col = VAR_df["Data Type"]
        VAR_df = VAR_df.transpose()
        VAR_df.columns = new_col
        VAR_df = VAR_df.iloc[2:,:]
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

        # if True plot
        if True:
            plot_var = model_data[0]
            # plt.figure()
            # plt.plot(df_forecast[plot_var + "_2d"], label='forecast')
            # plt.plot(test_diff_df[plot_var], label='actual')
            # plt.legend()
            plt.figure()
            plt.plot(df_forecast[plot_var + "_forecast"], label='forecast')
            plt.plot(test_df[plot_var], label='actual')
            plt.legend()
            plt.show()



# import data
panting_df = pd.read_csv("Clean Dataset Output/panting_timeseries.csv")
resting_df = pd.read_csv("Clean Dataset Output/resting_timeseries.csv")

# extract ordered list of animals and data type
cow_list = list(set(panting_df["Cow"]))
cow_list = sorted(cow_list)
#cow_list = ['8027107', '8022092', '8027476', '8032505', '8045911']
#cow_list = ['8027107']
data_type_list = list(set(panting_df["Data Type"]))
data_type_list = sorted(data_type_list)

df_list = [panting_df, resting_df]
headers = ["panting filtered", "resting filtered"]
# headers = ["resting filtered", "panting filtered", "resting raw", "panting raw"]
run_VAR(df_list, headers, cow_list, 1710, 96)


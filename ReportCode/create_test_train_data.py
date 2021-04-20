import sys
sys.path.insert(1, 'Source Code')

import pandas as pd
import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from matplotlib import pyplot as plt
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LSTM
from keras.layers import RepeatVector
from keras.layers import TimeDistributed, Dropout, MaxPooling1D, AveragePooling1D
from keras import regularizers
from keras import Input
from keras import Model
from keras.layers import Concatenate
from keras.optimizers import adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, TerminateOnNaN
from keras import models
import pickle
import random
from math import floor, ceil
from keras.initializers import Orthogonal

def create_test_train_data(cows, all_data, lags, start_iter = 201, end_iter = 1100):
    # define test/train range
    iter = range(start_iter, end_iter)

    # build test/train data
    X = []
    Y = []
    horizon = 24

    # iterate through each cow
    for cow in cows:
        # skip herd data
        if cow == 'All':
            continue

        print(cow)

        # extract panting data
        panting_df = all_data["panting"]
        panting_df = panting_df[panting_df["Cow"] == cow]

        resting_df = all_data["resting"]
        resting_df = resting_df[resting_df["Cow"] == cow]

        medium_activity_df = all_data["medium activity"]
        medium_activity_df = medium_activity_df[medium_activity_df["Cow"] == cow]

        herd_df = all_data["panting"]
        herd_df = herd_df[herd_df["Cow"] == "All"]

        weather_df = all_data['weather']
        weather_df = weather_df['HLI']

        # iterate through each sample
        for sample in iter:
            # add panting to x and y sample
            panting_df_sample = panting_df[panting_df["next_ts"]==sample]
            x_sample = [panting_df_sample[[i for i in range(-lags,0)]].values[0]]
            y_sample = [y for y in panting_df_sample[[i for i in range(0,horizon)]].values[0]]

            herd_df_sample = herd_df[herd_df["next_ts"] == sample]
            x_sample.append(herd_df_sample[[i for i in range(-lags,0)]].values[0])

            x_sample.append(weather_df.iloc[sample-lags-1:sample-1].values)

            resting_df_sample = resting_df[resting_df["next_ts"] == sample]
            x_sample.append(resting_df_sample[[i for i in range(-lags, 0)]].values[0])

            medium_activity_df_sample = medium_activity_df[medium_activity_df["next_ts"] == sample]
            x_sample.append(medium_activity_df_sample[[i for i in range(-lags, 0)]].values[0])

            # reshape in two stages to get desired behaviour
            x_sample = np.array(x_sample)
            a = x_sample.shape[1]
            b = x_sample.shape[0]
            x_sample = x_sample.reshape(a*b)
            x_sample = x_sample.reshape(a, b, order = 'F')

            # append
            X.append(x_sample)
            Y.append(y_sample)

    X = np.array(X)
    Y = np.array(Y)
    return X, Y

def normalise_data(train_x, train_y, test_x, test_y):
    # calculate standardise scalar on x_train and standardise
    std_train_x, scalar_x = normalise_x(train_x)

    # print mean values
    print(scalar_x.data_max_)
    print(scalar_x.data_min_)
    print(scalar_x.min_)
    print(scalar_x.scale_)
    print(scalar_x.data_range_)

    std_test_x, scalar_x = normalise_x(test_x, scalar_x)

    scalar_y = MinMaxScaler()
    scalar_y.max_ = scalar_x.data_max_[0]
    scalar_y.max_ = scalar_x.data_min_[0]
    scalar_y.min_ = scalar_x.min_[0]
    scalar_y.scale_ = scalar_x.scale_[0]
    scalar_y.data_range_ = scalar_x.data_range_[0]

    std_train_y = scalar_y.transform(train_y)
    std_test_y = scalar_y.transform(test_y)

    return std_train_x, std_train_y, std_test_x, std_test_y, scalar_y

def normalise_x(x_data, scalar=False):
    # create new data for standardisation
    new = []
    for i in range(0, len(x_data)):
        new.extend(x_data[i])

    # if no scalar given, fit one
    if not scalar:
        scalar = MinMaxScaler()
        scalar.fit(new)

    # transform data and reshape to original
    new = scalar.transform(new)
    new = np.array(new)
    x_new = new.reshape(x_data.shape[0], x_data.shape[1], x_data.shape[2])

    return x_new, scalar

def write_pickle(x_train, y_train, x_test, y_test, scalar_y, folder):
    with open(folder + 'x_train.pkl', 'wb') as f:
        pickle.dump(x_train, f)
    with open(folder + 'x_test.pkl', 'wb') as f:
        pickle.dump(x_test, f)
    with open(folder + 'y_train.pkl', 'wb') as f:
        pickle.dump(y_train, f)
    with open(folder + 'y_test.pkl', 'wb') as f:
        pickle.dump(y_test, f)
    with open(folder + 'scalar_y.pkl', 'wb') as f:
        pickle.dump(scalar_y, f)

def edit_num_lags(train_x, test_x, new_lag):
    train_x = reduce_lags(new_lag, train_x)
    test_x = reduce_lags(new_lag, test_x)
    return train_x, test_x

def reduce_lags(new_lag, l):
    l = list(l)
    for i in range(len(l)):
        l[i] = l[i][-new_lag:]
    return np.array(l)

def extract_forecast(x_values, y_values, num_cows):
    samples_per_cow = int(len(x_values) / num_cows)
    # print(len(x_values))
    # print(len(y_values))
    # print(samples_per_cow)
    new_input = []
    # extract next 24 hour forecast
    for cow_n in range(0, num_cows):
        for sample_n in range(0, samples_per_cow-24):
            # extract iter value
            i = sample_n + samples_per_cow * cow_n
            sample = []
            for j in range(-24,0):
                sample.append([x_values[i+24][j][2]])
            new_input.append(sample)

    # delete last 24 inputs
    for k in range(1, 25):
        x_values = np.delete(x_values, [j for j in range(samples_per_cow-k, len(x_values), samples_per_cow-k+1)], axis=0)
        y_values = np.delete(y_values, [j for j in range(samples_per_cow-k, len(y_values), samples_per_cow-k+1)], axis=0)

    return new_input, x_values, y_values

def add_forecast_input(train_x, train_y, test_x, test_y, num_cows):
    new_train_x, train_x, train_y = extract_forecast(train_x, train_y, num_cows)
    train_x_comb = [np.array(train_x), np.array(new_train_x)]

    new_test_x, test_x, test_y = extract_forecast(test_x, test_y, num_cows)
    test_x_comb = [np.array(test_x), np.array(new_test_x)]
    return train_x_comb, train_y, test_x_comb, test_y

def convert_to_univariate(train_x, test_x):
    train_x_new = []
    test_x_new = []
    train_x_old = train_x[0]
    test_x_old = test_x[0]
    # create new train x
    for sample in train_x_old:
        new_sample = []
        for time_step in sample:
            new_sample.append([time_step[0]])
        train_x_new.append(new_sample)
    # create new test x
    for sample in test_x_old:
        new_sample = []
        for time_step in sample:
            new_sample.append([time_step[0]])
        test_x_new.append(new_sample)

    test_x_new = [np.array(test_x_new), test_x[1]]
    train_x_new = [np.array(train_x_new), train_x[1]]

    return train_x_new, test_x_new

def save_test_train_data(lag, location, univariate = False, n_fold=0, num_cows=197):
    # set test and train values
    if n_fold:
        train_n = int(1455/6) * n_fold + 24 + 201
        test_n = train_n + int(1455/6) + 24
    else:
        train_n = 1100
        test_n = 1704

    print(train_n)
    print(test_n)

    # create test and train data from saved dataframes
    x_train, y_train = create_test_train_data(cow_list, all_data_dict, lag, 201, train_n) # original train (201, 1100)
    x_test, y_test = create_test_train_data(cow_list, all_data_dict, lag, train_n, test_n) # original test (1100, 1704)

    # normalise the data
    x_train, y_train, x_test, y_test, scalar_y = normalise_data(x_train, y_train, x_test, y_test)

    # add the next 24 hours of weather data to the input
    x_train, y_train, x_test, y_test = add_forecast_input(x_train, y_train, x_test, y_test, num_cows)

    print(len(y_train))
    print(len(y_test))

    # if univariate data is required, remove other series from input
    if univariate:
        x_train, x_test = convert_to_univariate(x_train, x_test)

    # save to file
    write_pickle(x_train, y_train, x_test, y_test, scalar_y, location)

# import state behaviour data
panting_model_df = pd.read_pickle("Deep Learning Data/panting model data.pkl")
resting_model_df = pd.read_pickle("Deep Learning Data/resting model data.pkl")
medium_activity_model_df = pd.read_pickle("Deep Learning Data/medium activity model data.pkl")

# import weather data and select measurements on the hour
weather_df = pd.read_csv("Weather.csv", encoding='utf-8')
weather_df = weather_df.iloc[[i for i in range(288,9163,6)]+[j for j in range(9164,10791,6)]]

# add all data to a single dictionary
all_data_dict = {"panting": panting_model_df, "resting": resting_model_df, "medium activity": medium_activity_model_df, "weather": weather_df}

# get cow list
cow_list = list(set(panting_model_df["Cow"]))
cow_list = sorted(cow_list)

# save n_fold cross validation test and train to file
for n in range(1,6):
    save_test_train_data(200, 'Deep Learning Data/Multivariate Lag 200/' + str(n) + '_fold/', n_fold = n)

# save_test_train_data(120, 'Deep Learning Data/Multivariate Lag 120/', univariate=True)
# save_test_train_data(120, 'Deep Learning Data/Univariate Lag 120/', univariate=True)
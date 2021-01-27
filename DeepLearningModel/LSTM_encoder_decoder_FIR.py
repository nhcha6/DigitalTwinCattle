import sys
sys.path.insert(1, '/Users/nicolaschapman/Documents/DigitalTwinCattle/DigitalTwinCattle/TrendsAndFiltering')
sys.path.insert(1, '/Users/nicolaschapman/Documents/DigitalTwinCattle/DigitalTwinCattle/Regressions')

import pandas as pd
import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LSTM
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.layers import Add
from keras.optimizers import adam
from keras.callbacks import EarlyStopping
from keras import models
import pickle
import random

def create_test_train_data(df_panting, df_resting, df_medium_activity, df_weather, cows, lags, horizon, test_train, run_time = False, num_cows = 198, inverse_flag = False):
    # define test or train
    if test_train == 'train':
        iter = range(201,1100)
    else:
        iter = range(1100,1704)

    # update iter for only a given time:
    if run_time:
        iter_new = []
        for j in iter:
            if j%24 == run_time:
                iter_new.append(j)
        iter = iter_new

    # build test/train data
    X = []
    Y = []
    inverse_list = []

    # extract herd data
    df_herd = df_panting[(df_panting["Cow"] == 'All') & (df_panting["Data Type"] == 'panting filtered')]
    df_weather = df_weather['HLI']

    # iterate through each cow
    cow_count = 0
    for cow in cows:
        # skip herd data
        if cow == 'All':
            continue

        if cow_count == num_cows:
            break

        print(cow)

        # extract panting data
        df_panting_cow = df_panting[(df_panting["Cow"] == cow) & (df_panting["Data Type"]=='panting filtered')]

        df_resting_cow = df_resting[(df_resting["Cow"] == cow) & (df_resting["Data Type"]=='resting filtered')]

        df_medium_activity_cow = df_medium_activity[(df_medium_activity["Cow"] == cow) & (df_medium_activity["Data Type"]=='medium activity filtered')]

        if inverse_flag:
            inverse_panting_cow = orig_panting_df[(orig_panting_df["Cow"] == cow) & (orig_panting_df["Data Type"]=='panting filtered')]

        for sample in iter:
            # add panting to x and y sample
            x_sample = [df_panting_cow[[str(i) for i in range(sample-lags, sample)]].values[0]]
            y_sample = [y for y in df_panting_cow[[str(i) for i in range(sample, sample+horizon)]].values[0]]

            x_sample.append(df_herd[[str(i) for i in range(sample-lags, sample)]].values[0])

            x_sample.append(df_weather.iloc[sample - lags - 1:sample - 1].values)

            x_sample.append(df_resting_cow[[str(i) for i in range(sample - lags, sample)]].values[0])

            x_sample.append(df_medium_activity_cow[[str(i) for i in range(sample - lags, sample)]].values[0])

            if inverse_flag:
                inverse_list.append(inverse_panting_cow[[str(i) for i in range(sample-2, sample)]].values[0])

            # reshape in two stages to get desired behaviour
            x_sample = np.array(x_sample)
            a = x_sample.shape[1]
            b = x_sample.shape[0]
            x_sample = x_sample.reshape(a * b)
            x_sample = x_sample.reshape(a, b, order='F')

            # append
            X.append(x_sample)
            Y.append(y_sample)

        cow_count += 1

    X = np.array(X)
    Y = np.array(Y)

    return X, Y, inverse_list

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


def build_model(train_x, train_y, test_x, test_y, batch_size, epochs, encoder_units, decoder_units, dense_neurons):
    # define hyper-parameters to be optimised
    verbose = 1
    loss = 'mse'
    optimiser = adam(clipvalue=1)
    activation = 'relu'
    callback = EarlyStopping(monitor='val_loss', patience=5)

    n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
    # reshape output into [samples, timesteps, features]
    train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))
    test_y = test_y.reshape((test_y.shape[0], test_y.shape[1], 1))

    # define model
    model = Sequential()
    model.add(LSTM(encoder_units, activation=activation, input_shape=(n_timesteps, n_features)))
    model.add(RepeatVector(n_outputs))
    model.add(LSTM(decoder_units, activation=activation, return_sequences=True))
    model.add(TimeDistributed(Dense(dense_neurons, activation=activation)))
    model.add(TimeDistributed(Dense(1)))
    model.compile(loss=loss, optimizer=optimiser)
    # fit network
    history = model.fit(train_x, train_y, validation_data=(test_x, test_y), epochs=epochs, batch_size=batch_size, verbose=verbose, callbacks=[callback])
    return model, history

def write_pickle(x_train, y_train, x_test, y_test, invert_test, scalar_y):
    with open('x_train.pkl', 'wb') as f:
        pickle.dump(x_train, f)
    with open('x_test.pkl', 'wb') as f:
        pickle.dump(x_test, f)
    with open('y_train.pkl', 'wb') as f:
        pickle.dump(y_train, f)
    with open('y_test.pkl', 'wb') as f:
        pickle.dump(y_test, f)
    with open('invert_test.pkl', 'wb') as f:
        pickle.dump(invert_test, f)
    with open('scalar_y.pkl', 'wb') as f:
        pickle.dump(scalar_y, f)

def read_pickle(folder):
    with open(folder + '/x_train.pkl', 'rb') as f:
        x_train = pickle.load(f)
    with open(folder + '/x_test.pkl', 'rb') as f:
        x_test = pickle.load(f)
    with open(folder + '/y_train.pkl', 'rb') as f:
        y_train = pickle.load(f)
    with open(folder + '/y_test.pkl', 'rb') as f:
        y_test = pickle.load(f)
    with open(folder + '/invert_test.pkl', 'rb') as f:
        invert_test = pickle.load(f)
    with open(folder + '/scalar_y.pkl', 'rb') as f:
        scalar_y = pickle.load(f)

    return x_train, y_train, x_test, y_test, invert_test, scalar_y

def train_from_original_data(lag, batch_size, epochs, encoder_units, decoder_units, dense_neurons, run_time=False, num_cows=198, diff=False):

    x_train, y_train, inverse_train = create_test_train_data(panting_df, resting_df, medium_activity_df, weather_df, cow_list, lag, 24, 'train', run_time, num_cows, diff)
    # print(x_train.shape)
    # print(y_train.shape)
    x_test, y_test,inverse_test = create_test_train_data(panting_df, resting_df, medium_activity_df, weather_df, cow_list, lag, 24, 'test', run_time, num_cows, diff)
    # print(x_test.shape)
    # print(y_test.shape)

    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)

    x_train, y_train, x_test, y_test, scalar_y = normalise_data(x_train, y_train, x_test, y_test)

    write_pickle(x_train, y_train, x_test, y_test, inverse_test, scalar_y)

    model, history = build_model(x_train, y_train, x_test, y_test, batch_size, epochs, encoder_units, decoder_units, dense_neurons)

    test_error(model, x_test, y_test, scalar_y, num_cows, inverse_test)
    # model.save('models/smallDiff')

def import_model(test_file, model_file):
    x_train, y_train, x_test, y_test, invert_test = read_pickle(test_file)

    x_train, x_test = edit_num_lags(x_train, x_test, 120)

    # scalar_y = MinMaxScaler()
    # scalar_y.max_ = 60.25546915
    # scalar_y.max_ = -7.45523883
    # scalar_y.min_ = 0.11010428
    # scalar_y.scale_ = 0.01476871
    # scalar_y.data_range_ = 67.71070798

    scalar_y = MinMaxScaler()
    scalar_y.max_ = 8.26615542
    scalar_y.max_ = -7.88232078
    scalar_y.min_ = 0.48811545
    scalar_y.scale_ = 0.06192535
    scalar_y.data_range_ = 16.1484762

    print("loading model")
    model = models.load_model(model_file)
    test_error(model, x_test, y_test, scalar_y, 3, invert_test)

def invert_differening(diff_seq, init):
    first_inv = [init[1]-init[0]]
    for i in range(len(diff_seq)):
        first_inv.append(first_inv[i] + diff_seq[i])

    second_inv = [init[0]]
    for i in range(len(first_inv)):
        second_inv.append(second_inv[i] + first_inv[i])

    return second_inv[2:]

def test_error(model, test_x, test_y, norm_y, num_cows=198, invert = []):
    y_pred = model.predict(test_x)

    # error calc
    false_pos = 0
    false_neg = 0
    total_pos = 0
    total_neg = 0
    hourly_errors = []
    daily_errors = []
    predictions = []


    # iter for doing only early morning analysis:
    iter = []
    for j in range(0, 119592, 604):
        for i in range(5, 604, 24):
            # iter.append(i - 2 + j)
            # iter.append(i - 1 + j)
            iter.append(i + j)
            # iter.append(i + 1 + j)
            # iter.append(i + 2 + j)


    # for i in iter:
    for sample_n in range(0, 604):
        freq_actual_dict = {}
        freq_forecast_dict = {}
        for cow_n in range(0,num_cows):
            # extract prediction
            i = sample_n + cow_n*604
            y_pred_orig = norm_y.inverse_transform(y_pred[i])
            test_y_i = test_y[i].reshape(-1,1)
            y_actual_orig = norm_y.inverse_transform(test_y_i)

            # plt.figure()
            # plt.plot(y_pred_orig, label='forecast')
            # plt.plot(y_actual_orig, label='actual')
            # plt.legend()

            # take difference if required
            if invert:
                y_actual_orig = invert_differening(y_actual_orig, invert[i])
                y_pred_orig = invert_differening(y_pred_orig, invert[i])

            error = mean_squared_error(y_actual_orig, y_pred_orig, squared=False)
            norm_error = error/max(y_actual_orig)

            # calculate frequencies
            freq_actual = sum(y_actual_orig)
            freq_forecast = sum(y_pred_orig)

            # error in frequency prediction
            daily_error = abs(freq_actual - freq_forecast)
            freq_actual_dict[cow_n] = freq_actual
            freq_forecast_dict[cow_n] = freq_forecast

            if max(y_actual_orig) > 1:
                hourly_errors.append(norm_error)
                daily_errors.append(daily_error)

            # calculate false positive and false negative above and below threshold
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

            # plt.figure()
            # print(error)
            # print(daily_error)
            # plt.plot(y_pred_orig, label='forecast')
            # plt.plot(y_actual_orig, label='actual')
            # plt.legend()
            # plt.show()

        # calculate top 20:
        freq_forecast_df = pd.DataFrame.from_dict(freq_forecast_dict, orient='index').sort_values(by=[0], ascending=False)
        freq_actual_df = pd.DataFrame.from_dict(freq_actual_dict, orient='index').sort_values(by=[0], ascending=False)
        top_20_forecast = set(freq_forecast_df.iloc[0:20, 0].index)
        top_20_actual = set(freq_actual_df.iloc[0:20, 0].index)
        top_20_predicted = [x for x in top_20_forecast if x in top_20_actual]
        predictions.append(len(top_20_predicted))

    print("\n")
    print("\n")
    print("\nMEAN HOURLY RMSE  TEST DATA")
    print(np.mean(hourly_errors))
    print("\nRMSE DAILY ERROR TEST DATA")
    print(np.mean(daily_errors))
    print("\nMEAN TOP 20 PREDICTED TEST DATA")
    print(np.mean(predictions))
    print("\nFALSE POS TEST DATA")
    print(false_pos/total_neg)
    print("\nFALSE NEG TEST DATA")
    print(false_neg/total_pos)
    print("\n")
    print("\n")

    return np.mean(hourly_errors), false_pos/total_neg, false_neg/total_pos

def random_hyper_search():
    lag_options = [48, 84, 120, 156, 192]
    encoder_units_options = [16, 32, 64, 128, 256]
    decoder_units_options = [16, 32, 64, 128, 256]
    dense_neurons_options = [32, 64, 128, 256, 512]
    batch_size_options = [128, 512, 2048, 4096, 8192]
    ignore_options = [0, 2, 4, 6, 8]

    hyperparams_dict = {'lag': [], 'encoder units': [], 'decoder units': [], 'dense neurons': [], 'batch_size': [],
                        'ignore_size': []}
    metric_dict = {'RMSE': [], 'false pos': [], 'false neg': [], 'epochs': []}

    for i in range(15):
        random.seed()

        index = random.randint(0, 4)
        lag = lag_options[index]
        hyperparams_dict['lag'].append(lag)

        index = random.randint(0, 4)
        encoder_units = encoder_units_options[index]
        hyperparams_dict['encoder units'].append(encoder_units)

        index = random.randint(0, 4)
        decoder_units = decoder_units_options[index]
        hyperparams_dict['decoder units'].append(decoder_units)

        index = random.randint(0, 4)
        dense_neurons = dense_neurons_options[index]
        hyperparams_dict['dense neurons'].append(dense_neurons)

        index = random.randint(0, 4)
        batch_size = batch_size_options[index]
        hyperparams_dict['batch_size'].append(batch_size)

        index = random.randint(0, 4)
        ignore_size = ignore_options[index]
        hyperparams_dict['ignore_size'].append(ignore_size)

        mean_RMSE, false_pos, false_neg, epochs = train_from_saved_data('normalised complete', lag=lag,
                                                                        batch_size=batch_size, epochs=1000,
                                                                        encoder_units=encoder_units,
                                                                        decoder_units=decoder_units,
                                                                        dense_neurons=dense_neurons, ignore_lags=ignore_size)

        metric_dict['RMSE'].append(mean_RMSE)
        metric_dict['false pos'].append(false_pos)
        metric_dict['false neg'].append(false_neg)
        metric_dict['epochs'].append(epochs)

    for param, param_list in hyperparams_dict.items():
        for metric, metric_list in metric_dict.items():
            plt.figure()
            plt.xlabel(param)
            plt.ylabel(metric)
            plt.plot(param_list, metric_list, 'ro')

    plt.show()


# import data
panting_df = pd.read_csv("Clean FIR Output/panting_timeseries.csv")
resting_df = pd.read_csv("Clean FIR Output/resting_timeseries.csv")
medium_activity_df = pd.read_csv("Clean FIR Output/medium activity_timeseries.csv")
weather_df = pd.read_csv("Weather.csv", encoding='utf-8')
weather_df = weather_df.iloc[[i for i in range(288,9163,6)]+[j for j in range(9164,10791,6)]]
orig_panting_df = panting_df.copy()

# extract ordered list of animals and data type
cow_list = list(set(panting_df["Cow"]))
cow_list = sorted(cow_list)
data_type_list = list(set(panting_df["Data Type"]))
data_type_list = sorted(data_type_list)

# create diff data
panting_df.iloc[:,2:] = panting_df.iloc[:,2:].diff(axis=1)
panting_df.iloc[:,2:] = panting_df.iloc[:,2:].diff(axis=1)

resting_df.iloc[:,2:] = resting_df.iloc[:,2:].diff(axis=1)
resting_df.iloc[:,2:] = resting_df.iloc[:,2:].diff(axis=1)

medium_activity_df.iloc[:,2:] = medium_activity_df.iloc[:,2:].diff(axis=1)
medium_activity_df.iloc[:,2:] = medium_activity_df.iloc[:,2:].diff(axis=1)

weather_df.iloc[:,2:] = weather_df.iloc[:,2:].diff(axis=0)
weather_df.iloc[:,2:] = weather_df.iloc[:,2:].diff(axis=0)


################### TRAIN FROM ORIGINAL PICKLE #########################

train_from_original_data(lag=120, batch_size=4096, epochs=1000, encoder_units=64, decoder_units=64, dense_neurons=128, diff=True)

################################################################################

########################## IMPORT OLD MODEL ############################

# import_model('small diff datatset', 'models/smallDiff')

###########################################################################

#################### OPTIMISATION TESTS #########################

# random_hyper_search()
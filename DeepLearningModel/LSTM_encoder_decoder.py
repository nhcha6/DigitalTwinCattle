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
from keras.layers import TimeDistributed, Dropout
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

def create_test_train_data(cows, all_data, lags, horizon, test_train, run_time = False, num_cows = 198, invert_diff = False):
    # define test or train
    if test_train == 'train':
        iter = range(213,1100)
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
    invert_diff_new = []


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
            # print(sample)
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

            # update invert_diff_new
            if invert_diff:
                invert_diff_new.append(invert_diff[cow_count*1504+sample-201])

        cow_count += 1


    X = np.array(X)
    Y = np.array(Y)
    return X, Y, invert_diff_new

def standardise_data(train_x, train_y, test_x, test_y):
    # calculate standardise scalar on x_train and standardise
    std_train_x, scalar_x = standardise_x(train_x)

    # print mean values
    print(scalar_x.mean_)
    print(scalar_x.var_)
    print(scalar_x.scale_)

    std_test_x, scalar_x = standardise_x(test_x, scalar_x)

    scalar_y = StandardScaler()
    scalar_y.mean_ = scalar_x.mean_[0]
    scalar_y.var_ = scalar_x.var_[0]
    scalar_y.scale_ = scalar_x.scale_[0]

    std_train_y = scalar_y.transform(train_y)
    std_test_y = scalar_y.transform(test_y)

    return std_train_x, std_train_y, std_test_x, std_test_y, scalar_y

def standardise_x(x_data, scalar=False):
    # create new data for standardisation
    new = []
    for i in range(0, len(x_data)):
        new.extend(x_data[i])

    # if no scalar given, fit one
    if not scalar:
        scalar = StandardScaler()
        scalar.fit(new)


    # transform data and reshape to original
    new = scalar.transform(new)
    new = np.array(new)
    x_new = new.reshape(x_data.shape[0], x_data.shape[1], x_data.shape[2])

    return x_new, scalar

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

def write_pickle(x_train, y_train, x_test, y_test, scalar_y):
    with open('x_train.pkl', 'wb') as f:
        pickle.dump(x_train, f)
    with open('x_test.pkl', 'wb') as f:
        pickle.dump(x_test, f)
    with open('y_train.pkl', 'wb') as f:
        pickle.dump(y_train, f)
    with open('y_test.pkl', 'wb') as f:
        pickle.dump(y_test, f)
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
    with open(folder + '/scalar_y.pkl', 'rb') as f:
        scalar_y = pickle.load(f)

    return x_train, y_train, x_test, y_test, scalar_y

def edit_num_lags(train_x, test_x, new_lag):
    train_x = reduce_lags(new_lag, train_x)
    test_x = reduce_lags(new_lag, test_x)
    return train_x, test_x

def reduce_lags(new_lag, l):
    l = list(l)
    for i in range(len(l)):
        l[i] = l[i][-new_lag:]
    return np.array(l)

def edit_ignore_lags(train_x, test_x, ignore_lag):
    train_x = ignore(ignore_lag, train_x)
    test_x = ignore(ignore_lag, test_x)
    return train_x, test_x

def ignore(ignore_lag, l):
    l = list(l)
    for i in range(len(l)):
        l[i] = l[i][0:-ignore_lag]
    return np.array(l)

def build_forecast_model(train_x, train_y, test_x, test_y, batch_size, epochs, encoder_units, decoder_units, dense_neurons, learning_rate, clipnorm, sample_weights=[]):
    # define hyper-parameters to be optimised
    verbose = 1
    loss = 'mse'
    optimiser = adam(learning_rate=learning_rate, clipnorm = clipnorm)
    activation = 'relu'
    # callback = EarlyStopping(monitor='val_loss', patience=10)
    filepath = "model checkpoints/batch_size" + str(batch_size) + "-{epoch:02d}.hdf5"
    callback_model = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=50)
    callback_nan = TerminateOnNaN()

    n_timesteps, n_features, n_outputs = train_x[0].shape[1], train_x[0].shape[2], train_y.shape[1]
    # reshape output into [samples, timesteps, features]
    train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))
    test_y = test_y.reshape((test_y.shape[0], test_y.shape[1], 1))

    # define model
    encoder_input = Input(shape=(n_timesteps, n_features))
    forecast_input = Input(shape=(24,1))
    l2_reg = regularizers.l2(0)
    # l2_reg = regularizers.l2(0.01)
    encoder_layer_1 = LSTM(encoder_units, activation=activation, kernel_regularizer=l2_reg, kernel_initializer=Orthogonal())
    encoder_hidden_output = encoder_layer_1(encoder_input)
    # encoder_layer_2 = LSTM(encoder_units, activation=activation, kernel_regularizer=l2_reg)
    # encoder_output = encoder_layer_2(encoder_hidden_output)
    decoder_input = RepeatVector(n_outputs)(encoder_hidden_output)
    decoder_input = Dropout(0.2)(decoder_input)
    decoder_input = Concatenate(axis=2)([decoder_input, forecast_input])
    # model.add(RepeatVector(n_outputs))
    decoder_layer = LSTM(decoder_units, activation=activation, return_sequences=True, kernel_regularizer=l2_reg, kernel_initializer=Orthogonal())
    decoder_output = decoder_layer(decoder_input)
    # model.add(LSTM(decoder_units, activation=activation, return_sequences=True))
    dense_input = Dropout(0.2)(decoder_output)
    dense_layer = TimeDistributed(Dense(dense_neurons, activation=activation))
    dense_output = dense_layer(dense_input)
    outputs = TimeDistributed(Dense(1))(dense_output)
    # model.add(TimeDistributed(Dense(dense_neurons, activation=activation)))
    # model.add(TimeDistributed(Dense(1)))

    model = Model(inputs=[encoder_input, forecast_input], outputs=outputs, name="model")
    # print(model.summary())
    model.compile(loss=loss, optimizer=optimiser)
    # fit network
    if sample_weights:
        history = model.fit(train_x, train_y, validation_data=(test_x, test_y), epochs=epochs, batch_size=batch_size, verbose=verbose, sample_weight=np.array(sample_weights), callbacks=[callback_model, callback_nan])
    else:
        history = model.fit(train_x, train_y, validation_data=(test_x, test_y), epochs=epochs, batch_size=batch_size, verbose=verbose, callbacks=[callback_model, callback_nan])
    return model, history

def build_original_model(train_x, train_y, test_x, test_y, batch_size, epochs, encoder_units, decoder_units, dense_neurons, sample_weights=[]):
    # define hyper-parameters to be optimised
    verbose = 1
    loss = 'mse'
    optimiser = adam(clipvalue=1)
    activation = 'relu'
    # callback = EarlyStopping(monitor='val_loss', patience=10)

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
    print(model.summary())
    model.compile(loss=loss, optimizer=optimiser)
    # fit network
    if sample_weights:
        history = model.fit(train_x, train_y, validation_data=(test_x, test_y), epochs=epochs, batch_size=batch_size, verbose=verbose, sample_weight=np.array(sample_weights))
    else:
        history = model.fit(train_x, train_y, validation_data=(test_x, test_y), epochs=epochs, batch_size=batch_size, verbose=verbose)
    return model, history

def calculate_sample_weights(y_test, constant, scalar_y):
    sample_weights = []
    y_test = scalar_y.inverse_transform(y_test)
    for y_sample in y_test:
        # y_sample = scalar_y.inverse_transform(y_sample)
        y_sample = y_sample.flatten()
        daily_freq = sum(y_sample)
        if daily_freq<100:
            weight = 1
        elif daily_freq<160:
            weight = 2*constant
        elif daily_freq<240:
            weight = 4*constant
        else:
            weight = 8*constant
        sample_weights.append(weight)
    return sample_weights

def calculate_sample_weights_new(y_test, bins, scalar_y, combine_flag):
    daily_frequency = []
    y_test = scalar_y.inverse_transform(y_test)
    for y_sample in y_test:
        # y_sample = scalar_y.inverse_transform(y_sample)
        y_sample = y_sample.flatten()
        daily_frequency.append(sum(y_sample))
    hist = np.histogram(daily_frequency, bins)

    # delete final half of bins
    if combine_flag:
        new_bins = np.delete(hist[1], [x for x in range(ceil(bins / 1.5), bins)])
        last_sum = sum(hist[0][floor(bins / 2) - 1:])
        new_freq = np.delete(hist[0], [x for x in range(ceil(bins / 1.5) - 1, bins)])
        new_freq = np.append(new_freq, last_sum)
        hist = (new_freq, new_bins)
        print(hist)
        bins = ceil(bins / 1.5)

    sample_weights = []
    for daily_freq in daily_frequency:
        for i in range(0,bins):
            if (daily_freq >= hist[1][i]) and daily_freq <= hist[1][i+1]:
                weight = len(daily_frequency)/(2*hist[0][i])
                sample_weights.append(weight)
    return sample_weights

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

def train_from_saved_data(file_name, lag, batch_size, epochs, encoder_units, decoder_units, dense_neurons, num_cows = 197, weights_flag=0, predict_lag = 0, horizon=24, learning_rate = 0.001, clipnorm=0.01, combine_bins = False, model_name = 'forecast'):
    # read in data
    x_train, y_train, x_test, y_test, scalar_y = read_pickle(file_name)
    invert_test = []

    print("\n")
    print("\n")
    print("lag: " + str(lag))
    print("batch_size: " + str(batch_size))
    print("encoder_units: " + str(encoder_units))
    print("decoder_units: " + str(decoder_units))
    print("dense_neurons: " + str(dense_neurons))
    print("epochs: " + str(epochs))
    print("clipnorm: " + str(clipnorm))
    print("learning rate: " + str(learning_rate))
    print("\n")
    print("\n")

    # remove first animal as it has a 3 day period of invalid data
    x_train = x_train[899:]
    y_train = y_train[899:]
    x_test = x_test[604:]
    y_test = y_test[604:]

    # reduce size for tests
    if num_cows != 197:
        x_train = x_train[0:num_cows*899]
        y_train = y_train[0:num_cows*899]
        x_test = x_test[0:num_cows*604]
        y_test = y_test[0:num_cows*604]

    # smaller prediction horizon
    if horizon != 24:
        y_train = y_train[:,0:horizon]
        y_test = y_test[:,0:horizon]

    if lag!=200:
        x_train, x_test = edit_num_lags(x_train, x_test, lag)

    # scalar_y = MinMaxScaler()
    # scalar_y.max_ = 61.24368378
    # scalar_y.max_ = -7.45523883
    # scalar_y.min_ = 0.10852046
    # scalar_y.scale_ = 0.01455627
    # scalar_y.data_range_ = 68.69892261

    if model_name == 'forecast':
        x_train, y_train, x_test, y_test = add_forecast_input(x_train, y_train, x_test, y_test, num_cows)
        # print(x_train[0][101])
        # print(x_train[1][100])

    sample_weights = []
    if weights_flag:
        # sample_weights = calculate_sample_weights(y_train, weights_flag, scalar_y)
        sample_weights = calculate_sample_weights_new(y_train, weights_flag, scalar_y, combine_bins)

    if model_name == 'forecast':
        model, history = build_forecast_model(x_train, y_train, x_test, y_test, batch_size, epochs, encoder_units, decoder_units, dense_neurons, learning_rate, clipnorm, sample_weights)
    elif model_name == 'original':
        model, history = build_original_model(x_train, y_train, x_test, y_test, batch_size, epochs, encoder_units, decoder_units, dense_neurons, sample_weights)

    # model.save('models/' + file_name)
    mean_RMSE, false_pos, false_neg, daily_RMSE, mean_pred = test_error(model, x_test, y_test, scalar_y, num_cows, invert_test, model_type=model_name)

    if predict_lag:
        test_error(model, x_test, y_test, scalar_y, num_cows, invert_test, predict_lag, model_type=model_name)

    return mean_RMSE, false_pos, false_neg, epochs, mean_RMSE, mean_pred, model

def train_from_original_data(lag, batch_size, epochs, encoder_units, decoder_units, dense_neurons, run_time=False, num_cows=197):
    x_train, y_train, invert_train = create_test_train_data(cow_list, all_data_dict, lag, 24, 'train', run_time, num_cows, invert_diff)
    # print(x_train.shape)
    # print(y_train.shape)
    x_test, y_test,invert_test = create_test_train_data(cow_list, all_data_dict, lag, 24, 'test', run_time, num_cows, invert_diff)
    # print(x_test.shape)
    # print(y_test.shape)

    x_train, y_train, x_test, y_test, scalar_y = normalise_data(x_train, y_train, x_test, y_test)
    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)

    write_pickle(x_train, y_train, x_test, y_test, invert_test)

    model, history = build_model(x_train, y_train, x_test, y_test, batch_size, epochs, encoder_units, decoder_units, dense_neurons)

    test_error(model, x_test, y_test, scalar_y, num_cows, invert_test)
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

def test_error(model, test_x, test_y, norm_y, num_cows=197, invert_test = [],y_prev=0, model_type='forecast'):
    y_pred = model.predict(test_x)

    # only care about original test value for assessing error.
    if model_type=='forecast':
        test_x = test_x[0]

    samples_per_cow = int(test_x.shape[0]/num_cows)

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
    for j in range(0, 119592, samples_per_cow):
        for i in range(5, 604, 24):
            # iter.append(i - 2 + j)
            # iter.append(i - 1 + j)
            iter.append(i + j)
            # iter.append(i + 1 + j)
            # iter.append(i + 2 + j)

    # for i in iter:
    for sample_n in range(y_prev, samples_per_cow):
        freq_actual_dict = {}
        freq_forecast_dict = {}
        for cow_n in range(0,num_cows):
            # extract prediction
            i = sample_n + cow_n*samples_per_cow
            y_pred_orig = norm_y.inverse_transform(y_pred[i])
            test_y_i = test_y[i].reshape(-1,1)
            y_actual_orig = norm_y.inverse_transform(test_y_i)

            # plt.figure()
            # plt.plot(y_pred_orig, label='forecast')
            # plt.plot(y_actual_orig, label='actual')
            # plt.legend()

            # take difference if required
            if invert_test:
                y_actual_orig = invert_differening(y_actual_orig, invert_test[i])
                y_pred_orig = invert_differening(y_pred_orig, invert_test[i])

            # adjust prediction so that it only uses the first 18 samples, and uses the actually recorder previous 6
            # to make 24 hours.
            if y_prev:
                # extract actual previous x values
                y_prev_actual = test_y[i-y_prev].reshape(-1,1)
                y_prev_actual = norm_y.inverse_transform(y_prev_actual)
                y_prev_actual = y_prev_actual[0:6]

                # extract available previous x values
                x_test_i = test_x[i]
                y_prev_known = []
                for j in range(-y_prev, 0):
                    y_prev_known.append([x_test_i[j][0]])
                y_prev_known = norm_y.inverse_transform(y_prev_known)

                y_actual_orig = np.concatenate((y_prev_actual, y_actual_orig[0:-y_prev]))
                y_pred_orig = np.concatenate((y_prev_known, y_pred_orig[0:-y_prev]))

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

            if False:
                plt.figure()
                print(error)
                print(daily_error)
                plt.plot(y_pred_orig, label='forecast')
                plt.plot(y_actual_orig, label='actual')
                plt.legend()
                plt.show()

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

    return np.mean(hourly_errors), false_pos/total_neg, false_neg/total_pos, np.mean(daily_errors), np.mean(predictions)

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
                                                                        dense_neurons=dense_neurons)

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

def random_hyper_optimisation():
    # params = [120, 409, 5, 64, 64, 128, 5]
    params = [200, 128, 80, 128, 64, 64, 7]

    mean_RMSE, false_pos, false_neg, epochs, daily_RMSE, mean_pred, model = train_from_saved_data('normalised complete',
                                                                                           lag=params[0],
                                                                                           batch_size=params[1],
                                                                                           epochs=params[2],
                                                                                           encoder_units=params[3],
                                                                                           decoder_units=params[4],
                                                                                           dense_neurons=params[5],
                                                                                           weights_flag=params[6],
                                                                                           predict_lag=6)
    error_metric = mean_RMSE

    while(True):
        random.seed()
        new_params = params.copy()
        for i in range(3):
            rand_index = random.randint(1,6)
            current = params[rand_index]
            rand_change = random.randint(ceil(-current/2), ceil(current/2))
            if rand_change > 0:
                rand_change = rand_change*2
            new_params[rand_index] = current + rand_change

        # # adjustment for lag above current data max
        # if new_params[0]>200:
        #     new_params[0] = 200

        try:
            mean_RMSE, false_pos, false_neg, epochs, daily_RMSE, mean_pred, model = train_from_saved_data('normalised complete', lag=new_params[0],
                                                                                batch_size=new_params[1], epochs=new_params[2],
                                                                                encoder_units=new_params[3],
                                                                                decoder_units=new_params[4],
                                                                                dense_neurons=new_params[5],
                                                                                weights_flag=new_params[6],
                                                                                predict_lag=6)
            if mean_RMSE < error_metric:
                print('\n\nLOWER ERROR: NEW BASELINE MODEL\n\n')
                error_metric = mean_RMSE
                params = new_params
                model.save('models/optimal_model')

        except:
            print('error')
            continue

# import state behaviour data
panting_model_df = pd.read_pickle("Model Data/panting model data.pkl")
resting_model_df = pd.read_pickle("Model Data/resting model data.pkl")
medium_activity_model_df = pd.read_pickle("Model Data/medium activity model data.pkl")

# import weather data
weather_df = pd.read_csv("Weather.csv", encoding='utf-8')
weather_df = weather_df.iloc[[i for i in range(288,9163,6)]+[j for j in range(9164,10791,6)]]

# data dict
all_data_dict = {"panting": panting_model_df, "resting": resting_model_df, "medium activity": medium_activity_model_df, "weather": weather_df}
invert_diff = []

# get cow list
cow_list = list(set(panting_model_df["Cow"]))
cow_list = sorted(cow_list)

#################### TRAIN FROM NORMALISED SAVED DATA #########################
#
# train_from_saved_data('normalised complete', lag=120, batch_size=409, epochs=10, encoder_units=64, decoder_units=64, dense_neurons=128, num_cows=10, weights_flag=7, predict_lag=6, model_name='original')
# train_from_saved_data('normalised complete', lag=120, batch_size=16, epochs=10, encoder_units=32, decoder_units=32, dense_neurons=16, num_cows=3, weights_flag=7, predict_lag=6, model_name='forecast')
# train_from_saved_data('normalised complete', lag=120, batch_size=32, epochs=50, encoder_units=16, decoder_units=32, dense_neurons=24, weights_flag=7, horizon=24, predict_lag=6, learning_rate=0.1, clipnorm=0.001, num_cows=5, model_name='forecast')

# train_from_saved_data('normalised complete', lag=120, batch_size=128, epochs=20, encoder_units=87, decoder_units=32, dense_neurons=64, weights_flag=7, horizon = 24, predict_lag=6, model_name='original')
# train_from_saved_data('normalised complete', lag=120, batch_size=128, epochs=20, encoder_units=32, decoder_units=32, dense_neurons=16, weights_flag=7, horizon = 24, predict_lag=6, model_name='forecast')

# train_from_saved_data('normalised complete', lag=120, batch_size=2616, epochs=20, encoder_units=87, decoder_units=32, dense_neurons=64, weights_flag=5, horizon = 24, predict_lag=6)
#################################################################################

################### TRAIN FROM ORIGINAL PICKLE #########################

# train_from_original_data(lag=120, batch_size=250, epochs=12, encoder_units=100, decoder_units=100, dense_neurons=100, run_time=False, num_cows=3)

################################################################################

########################## IMPORT OLD MODEL ############################

# import_model('small diff datatset', 'models/smallDiff')

###########################################################################

#################### OPTIMISATION TESTS #########################

# random_hyper_search()

# random_hyper_optimisation()
#
# best_RMSE = 1000
# batch_dict = {512: 0.0005, 256: 0.0005, 128: 0.0002, 64: 0.0001, 32: 0.00005}
# for batch_size, lr in batch_dict.items():
#     while(True):
#         try:
#             mean_RMSE, false_pos, false_neg, epochs, daily_RMSE, mean_pred, model = train_from_saved_data('normalised complete',
#                                                                                                       lag=120,
#                                                                                                       batch_size=batch_size,
#                                                                                                       epochs=300,
#                                                                                                       encoder_units=32,
#                                                                                                       decoder_units=64,
#                                                                                                       dense_neurons=48,
#                                                                                                       weights_flag=7,
#                                                                                                       horizon=24,
#                                                                                                       learning_rate=lr,
#                                                                                                       clipnorm=0.001,
#                                                                                                       predict_lag=6,
#                                                                                                       model_name='forecast')
#
#             if mean_RMSE<best_RMSE:
#                 best_RMSE = mean_RMSE
#                 model.save('models/optimal_model')
#
#             break
#
#         except:
#             print("Error")
#             lr = lr/2


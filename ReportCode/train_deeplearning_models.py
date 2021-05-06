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


def build_model_init_states(train_x, train_y, test_x, test_y, batch_size, epochs, encoder_units, decoder_units, dense_neurons, learning_rate, clipnorm, sample_weights=[], test_name='forecast'):
    # define hyper-parameters
    verbose = 1
    loss = 'mse'
    optimiser = adam(learning_rate=learning_rate, clipnorm = clipnorm, decay=learning_rate/200)
    activation = 'relu'

    # callback = EarlyStopping(monitor='val_loss', patience=10)
    filepath = "LSTM Models/Current Test/" + test_name + "-batch_size" + str(batch_size) + "-{epoch:02d}.hdf5"
    callback_model = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=50)
    callback_nan = TerminateOnNaN()

    # extract shape of input and output
    n_timesteps, n_features, n_outputs = train_x[0].shape[1], train_x[0].shape[2], train_y.shape[1]

    # reshape output into [samples, timesteps, features]
    train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))
    test_y = test_y.reshape((test_y.shape[0], test_y.shape[1], 1))

    # define model
    encoder_input = Input(shape=(n_timesteps, n_features))
    forecast_input = Input(shape=(24,1))

    encoder_layer_1 = LSTM(encoder_units, activation=activation, kernel_initializer=Orthogonal(), return_state=True)
    encoder_output, state_h, state_c = encoder_layer_1(encoder_input)
    encoder_states = [state_h, state_c]
    decoder_input = RepeatVector(n_outputs)(encoder_output)
    decoder_input = Dropout(0.2)(decoder_input)
    decoder_input = Concatenate(axis=2)([decoder_input, forecast_input])
    decoder_layer = LSTM(decoder_units, activation=activation, return_sequences=True, kernel_initializer=Orthogonal())
    decoder_output = decoder_layer(decoder_input, initial_state=encoder_states)
    dense_input = Dropout(0.2)(decoder_output)
    dense_layer = TimeDistributed(Dense(dense_neurons, activation=activation))
    dense_output = dense_layer(dense_input)
    outputs = TimeDistributed(Dense(1))(dense_output)

    model = Model(inputs=[encoder_input, forecast_input], outputs=outputs, name="model")
    print(model.summary())
    model.compile(loss=loss, optimizer=optimiser)
    # fit network
    if sample_weights:
        model.fit(train_x, train_y, validation_data=(test_x, test_y), epochs=epochs, batch_size=batch_size, verbose=verbose, sample_weight=np.array(sample_weights), callbacks=[callback_model, callback_nan])
    else:
        model.fit(train_x, train_y, validation_data=(test_x, test_y), epochs=epochs, batch_size=batch_size, verbose=verbose, callbacks=[callback_model, callback_nan])

    return model

def build_model(train_x, train_y, test_x, test_y, batch_size, epochs, encoder_units, decoder_units, dense_neurons, learning_rate, clipnorm, sample_weights=[], test_name='forecast'):
    # define hyper-parameters
    verbose = 1
    loss = 'mse'
    optimiser = adam(learning_rate=learning_rate, clipnorm = clipnorm, decay=learning_rate/200)
    activation = 'relu'

    # callback = EarlyStopping(monitor='val_loss', patience=10)
    filepath = "LSTM Models/Current Test/" + test_name + "-batch_size" + str(batch_size) + "-{epoch:02d}.hdf5"
    callback_model = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=50)
    callback_nan = TerminateOnNaN()

    # extract shape of input and output
    n_timesteps, n_features, n_outputs = train_x[0].shape[1], train_x[0].shape[2], train_y.shape[1]

    # reshape output into [samples, timesteps, features]
    train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))
    test_y = test_y.reshape((test_y.shape[0], test_y.shape[1], 1))

    # define model
    encoder_input = Input(shape=(n_timesteps, n_features))
    forecast_input = Input(shape=(24,1))

    encoder_layer_1 = LSTM(encoder_units, activation=activation, kernel_initializer=Orthogonal())
    encoder_hidden_output = encoder_layer_1(encoder_input)
    decoder_input = RepeatVector(n_outputs)(encoder_hidden_output)
    decoder_input = Dropout(0.2)(decoder_input)
    decoder_input = Concatenate(axis=2)([decoder_input, forecast_input])
    decoder_layer = LSTM(decoder_units, activation=activation, return_sequences=True, kernel_initializer=Orthogonal())
    decoder_output = decoder_layer(decoder_input)
    dense_input = Dropout(0.2)(decoder_output)
    dense_layer = TimeDistributed(Dense(dense_neurons, activation=activation))
    dense_output = dense_layer(dense_input)
    outputs = TimeDistributed(Dense(1))(dense_output)

    model = Model(inputs=[encoder_input, forecast_input], outputs=outputs, name="model")
    # print(model.summary())
    model.compile(loss=loss, optimizer=optimiser)
    # fit network
    if sample_weights:
        model.fit(train_x, train_y, validation_data=(test_x, test_y), epochs=epochs, batch_size=batch_size, verbose=verbose, sample_weight=np.array(sample_weights), callbacks=[callback_model, callback_nan])
    else:
        model.fit(train_x, train_y, validation_data=(test_x, test_y), epochs=epochs, batch_size=batch_size, verbose=verbose, callbacks=[callback_model, callback_nan])

    return model

def edit_num_lags(train_x, test_x, new_lag):
    train_x[0] = reduce_lags(new_lag, train_x[0])
    test_x[0] = reduce_lags(new_lag, test_x[0])
    return train_x, test_x

def reduce_lags(new_lag, l):
    l = list(l)
    for i in range(len(l)):
        l[i] = l[i][-new_lag:]
    return np.array(l)

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

def calculate_sample_weights(y_test, bins, scalar_y):
    daily_frequency = []
    y_test = scalar_y.inverse_transform(y_test)
    for y_sample in y_test:
        # y_sample = scalar_y.inverse_transform(y_sample)
        y_sample = y_sample.flatten()
        daily_frequency.append(sum(y_sample))
    hist = np.histogram(daily_frequency, bins)

    sample_weights = []
    for daily_freq in daily_frequency:
        for i in range(0,bins):
            if (daily_freq >= hist[1][i]) and daily_freq <= hist[1][i+1]:
                weight = len(daily_frequency)/(2*hist[0][i])
                sample_weights.append(weight)
    return sample_weights


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


def convert_to_bivariate(train_x, test_x):
    train_x_new = []
    test_x_new = []
    train_x_old = train_x[0]
    test_x_old = test_x[0]
    # create new train x
    for sample in train_x_old:
        new_sample = []
        for time_step in sample:
            new_sample.append([time_step[0], time_step[2]])
        train_x_new.append(new_sample)
    # create new test x
    for sample in test_x_old:
        new_sample = []
        for time_step in sample:
            new_sample.append([time_step[0], time_step[2]])
        test_x_new.append(new_sample)

    test_x_new = [np.array(test_x_new), test_x[1]]
    train_x_new = [np.array(train_x_new), train_x[1]]

    return train_x_new, test_x_new

def train_from_saved_data(file_name, lag, batch_size, epochs, encoder_units, decoder_units, dense_neurons, weights_flag=0, learning_rate = 0.001, num_cows = 197, test_name = 'multivariate'):
    # read in data
    x_train, y_train, x_test, y_test, scalar_y = read_pickle(file_name)

    print("\n")
    print("\n")
    print("lag: " + str(lag))
    print("batch_size: " + str(batch_size))
    print("encoder_units: " + str(encoder_units))
    print("decoder_units: " + str(decoder_units))
    print("dense_neurons: " + str(dense_neurons))
    print("epochs: " + str(epochs))
    print("learning rate: " + str(learning_rate))
    print("\n")
    print("\n")

    # reduce size for tests
    if num_cows != 197:
        x_train = [x_train[0][0:num_cows * 875], x_train[1][0:num_cows * 875]]
        y_train = y_train[0:num_cows * 875]
        x_test = [x_test[0][0:num_cows * 604], x_test[1][0:num_cows * 604]]
        y_test = y_test[0:num_cows * 604]

    # reduce number of lags
    if lag!=200:
        x_train, x_test = edit_num_lags(x_train, x_test, lag)

    # change to uni or bivariate data
    if test_name == 'univariate':
        x_train, x_test = convert_to_univariate(x_train, x_test)
    if test_name == 'bivariate':
        x_train, x_test = convert_to_bivariate(x_train, x_test)

    # change train/test size for cross validation
    # x_train, y_train, x_test, y_test = change_test_train_ratio(x_train, y_train, x_test, y_test, num_cows=197)

    sample_weights = []
    if weights_flag:
        sample_weights = calculate_sample_weights(y_train, weights_flag, scalar_y)

    # for i in range(1000):
    #     plt.figure()
    #     plt.plot(x_train[0][i])
    #     plt.title('lagged data')
    #
    #     plt.figure()
    #     plt.plot(x_train[1][i])
    #     plt.title('forecast weather')
    #
    #     plt.figure()
    #     plt.plot(y_train[i])
    #     plt.title('future panting')
    #     plt.show()

    model = build_model(x_train, y_train, x_test, y_test, batch_size, epochs, encoder_units, decoder_units, dense_neurons, learning_rate, 0.5, sample_weights, test_name=test_name)
    return model

def change_test_train_ratio(x_train, y_train, x_test, y_test, train_size = 500, num_cows=197):
    train_i = []
    test_i = []

    for cow_n in range(0, num_cows):
        for day_n in range(0, 875):
            if day_n < train_size:
                train_i.append(cow_n * 875 + day_n)
            else:
                test_i.append(cow_n * 875 + day_n)

    x_test = [np.concatenate((x_test[0], x_train[0][test_i])), np.concatenate((x_test[1], x_train[1][test_i]))]
    y_test = np.concatenate((y_test, y_train[test_i]))
    x_train = [x_train[0][train_i], x_train[1][train_i]]
    y_train = y_train[train_i]

    return x_train, y_train, x_test, y_test

def grid_search(batch_dict, saved_data, test_name):
    # loop through and test each model
    for batch_size, lr in batch_dict.items():
        train_from_saved_data(file_name=saved_data, lag=120, batch_size=batch_size, epochs=300, encoder_units=32,
                                                                                                      decoder_units=64,
                                                                                                      dense_neurons=48,
                                                                                                      weights_flag=7,
                                                                                                      learning_rate=lr,
                                                                                                      test_name=test_name)

def k_fold_test(batch_dict, saved_data):
    # loop through and test each model
    for n in range(1,6):
        n_fold = str(n) + '_fold'
        for batch_size, lr in batch_dict.items():
            n_fold_file = saved_data + '/' + n_fold
            train_from_saved_data(file_name=n_fold_file, lag=120, batch_size=batch_size, epochs=150, encoder_units=32,
                                                                                                          decoder_units=64,
                                                                                                          dense_neurons=48,
                                                                                                          weights_flag=7,
                                                                                                          learning_rate=lr,
                                                                                                          test_name='bivariate')

# Define batch size and learning rate for univariate test
# batch_dict = {512:0.0005, 256: 0.0005, 128: 0.0005, 64: 0.0005}
# grid_search(batch_dict, 'Deep Learning Data/Univariate Lag 120', 'univariate')

# Define batch size and learning rate for multivariate test
# batch_dict = {512:0.0005, 256: 0.0005, 128: 0.0002, 64: 0.00005}
# grid_search(batch_dict, 'Deep Learning Data/Multivariate Lag 120', '')

# run no sample weight test
# model = train_from_saved_data(file_name='Deep Learning Data/Univariate Lag 120', lag=120, batch_size=64, epochs=10, encoder_units=32, decoder_units=64,
#                                                                                                       dense_neurons=48,
#                                                                                                       weights_flag=0,
#                                                                                                       learning_rate=0.0005,
#                                                                                                       test_name='no_weight')
# model.save('LSTM Models/No Weight Tests/no_weights_model.hdf5')

# run k fold cross validation test
# batch_dict = {256: 0.0002, 128: 0.0001, 64: 0.00002}
# batch_dict = {256: 0.0005, 128: 0.0002, 64: 0.00005}
batch_dict = {256: 0.0005, 128: 0.0005, 64: 0.0005}
k_fold_test(batch_dict, 'Deep Learning Data/Multivariate Lag 200')
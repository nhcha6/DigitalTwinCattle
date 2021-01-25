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
from keras.optimizers import adam
import pickle

def create_test_train_data(input_data, cows, all_data, lags, horizon, test_train):
    # define test or train
    if test_train == 'train':
        iter = range(201,1100)
    else:
        iter = range(1100,1704)

    # build test/train data
    X = []
    Y = []
    # iterate through each cow
    cow_count = 0
    for cow in cows:
        cow_count+=1
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

        if cow_count == 5:
            break

    X = np.array(X)
    Y = np.array(Y)
    return X, Y

def standardise_data(train_x, train_y, test_x, test_y):
    # calculate standardise scalar on x_train and standardise
    std_train_x, scalar_x = standardise_x(train_x)

    # print mean values
    print(scalar_x.mean_)
    print(scalar_x.var_)

    std_test_x, scalar_x = standardise_x(test_x, scalar_x)

    scalar_y = StandardScaler()
    scalar_y.mean_ = scalar_x.mean_[0]
    scalar_y.var_ = scalar_x.var_[0]
    scalar_y.scale_ = scalar_x.scale_[0]

    std_train_y = scalar_y.transform(train_y)
    std_test_y = scalar_y.transform(test_y)

    return std_train_x, std_train_y, std_test_x, std_test_y

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

def write_pickle(x_train, y_train, x_test, y_test):
    with open('x_train.pkl', 'wb') as f:
        pickle.dump(x_train, f)
    with open('x_test.pkl', 'wb') as f:
        pickle.dump(x_test, f)
    with open('y_train.pkl', 'wb') as f:
        pickle.dump(y_train, f)
    with open('y_test.pkl', 'wb') as f:
        pickle.dump(y_test, f)

def read_pickle():
    with open('x_train.pkl', 'rb') as f:
        x_train = pickle.load(f)
    with open('x_test.pkl', 'rb') as f:
        x_test = pickle.load(f)
    with open('y_train.pkl', 'rb') as f:
        y_train = pickle.load(f)
    with open('y_test.pkl', 'rb') as f:
        y_test = pickle.load(f)

    return x_train, y_train, x_test, y_test

# train the model
def build_model(train_x, train_y, test_x, test_y):
    # define hyper-parameters to be otpimised
    verbose, epochs, batch_size = 1, 20, 128
    encoder_units = 50
    decoder_units = 50
    dense_neurons = 100
    loss = 'mse'
    optimiser = adam(clipvalue=5)
    activation = 'relu'

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
    history = model.fit(train_x, train_y, validation_data=(test_x, test_y), epochs=epochs, batch_size=batch_size, verbose=verbose)
    return model, history

def test_error(model, test_x, test_y, norm_y):
    y_pred = model.predict(test_x)
    for i in range(len(y_pred)):
        # y_pred_i = y_pred[i].reshape(24)
        y_pred_orig = norm_y.inverse_transform(y_pred[i])
        test_y_i = test_y[i].reshape(-1,1)
        y_actual_orig = norm_y.inverse_transform(test_y_i)
        error = mean_squared_error(y_actual_orig, y_pred_orig)
        norm_error = error/max(y_actual_orig)
        print(norm_error)
        plt.figure()
        plt.plot(y_pred_orig, label='forecast')
        plt.plot(y_actual_orig, label='actual')
        plt.legend()
        plt.show()



# import state behaviour data
panting_model_df = pd.read_pickle("Model Data/panting model data.pkl")
resting_model_df = pd.read_pickle("Model Data/resting model data.pkl")
medium_activity_model_df = pd.read_pickle("Model Data/medium activity model data.pkl")

# import weather data
weather_df = pd.read_csv("Weather.csv", encoding='utf-8')
weather_df = weather_df.iloc[[i for i in range(288,9163,6)]+[j for j in range(9164,10791,6)]]

# # normalise data
# print(panting_model_df.head())
# scalar = StandardScaler()
# panting_values = np.array(panting_model_df.values)
# shape = panting_values.shape
# col = panting_model_df.columns
# flat = panting_values.flatten()
# flat_std = scalar.fit_transform(flat)
# std_panting_values = flat_std.reshape(shape)
# print(scalar.mean_)
# std_panting_model_df = pd.DataFrame(std_panting_values, columns=col)
# print(std_panting_model_df.head())


# data dict
all_data_dict = {"panting": panting_model_df, "resting": resting_model_df, "medium activity": medium_activity_model_df, "weather": weather_df}

# select input data
model = ["resting", "medium activity", "HLI", "herd"]

# get cow list
cow_list = list(set(panting_model_df["Cow"]))
cow_list = sorted(cow_list)

x_train, y_train = create_test_train_data(model, cow_list, all_data_dict, 120, 12, 'train')
# print(x_train.shape)
# print(y_train.shape)
x_test, y_test = create_test_train_data(model, cow_list, all_data_dict, 120, 12, 'test')
# print(x_test.shape)
# print(y_test.shape)

x_train, y_train, x_test, y_test, scalar_y = normalise_data(x_train, y_train, x_test, y_test)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

# write_pickle(x_train,y_train, x_test, y_test)

# x_train,y_train, x_test, y_test = read_pickle()

scalar_y = MinMaxScaler()
scalar_y.max_ = 55.99594951
scalar_y.max_ = -7.08498508
scalar_y.min_ = 0.11231579
scalar_y.scale_ = 0.01585265
scalar_y.data_range_ = 63.08093459

model, history = build_model(x_train, y_train, x_test, y_test)
test_error(model, x_test, y_test, scalar_y)
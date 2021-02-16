import pandas as pd
import numpy as np
from math import sqrt, floor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn import svm
from matplotlib import pyplot as plt
from matplotlib import pyplot as plt
from keras.layers import Dense, Dropout
from keras import Input
from keras import Model
from keras.optimizers import adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, TerminateOnNaN
from keras import models
import pickle

def import_model(test_file, model_file):
    x_train, y_train, x_test, y_test, scalar_y = read_pickle(test_file)

    print("Loading Model")
    model = models.load_model(model_file)

    return model, x_train, y_train, x_test, y_test, scalar_y

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

def new_test_train(test_x, test_y, num_cows = 197):
    samples_per_cow = int(test_x[0].shape[0] / num_cows)
    cutoff = floor(samples_per_cow/3)

    train_x_lags = []
    train_x_weather = []
    train_y_new = []
    for cow_n in range(0, num_cows):
        for sample_n in range(0, cutoff):
            i = sample_n + cow_n * samples_per_cow
            train_x_lags.append(test_x[0][i])
            train_x_weather.append(test_x[1][i])
            train_y_new.append(test_y[i])

    test_x_lags = []
    test_x_weather = []
    test_y_new = []
    for cow_n in range(0, num_cows):
        for sample_n in range(cutoff, samples_per_cow):
            i = sample_n + cow_n * samples_per_cow
            test_x_lags.append(test_x[0][i])
            test_x_weather.append(test_x[1][i])
            test_y_new.append(test_y[i])

    train_x_post = [np.array(train_x_lags), np.array(train_x_weather)]
    train_y_post = np.array(train_y_new)
    test_x_post = [np.array(test_x_lags), np.array(test_x_weather)]
    test_y_post = np.array(test_y_new)

    # train_x_post = [np.array(test_x[0][0:38021]), np.array(test_x[1][0:38021])]
    # train_y_post = np.array(test_y[0:38021])
    # test_x_post = [np.array(test_x[0][38021:]), np.array(test_x[1][38021:])]
    # test_y_post = np.array(test_y[38021:])
    return train_x_post, train_y_post, test_x_post, test_y_post

def add_max_y(y):
    y_new = []
    for y_sample in y:
        y_new.append([max(y_sample)])
    return np.array(y_new)

def calculate_threshold_y(y, scalar_y, thresh):
    y_new = []
    y = scalar_y.inverse_transform(y)
    for y_sample in y:
        # print(y_sample)
        if max(y_sample)>thresh:
            y_new.append([1,0])
        else:
            y_new.append([0,1])
    return np.array(y_new)

def calculate_threshold_svm(y, scalar_y, thresh):
    y_new = []
    y = scalar_y.inverse_transform(y)
    for y_sample in y:
        # print(y_sample)
        if max(y_sample)>thresh:
            y_new.append(1)
        else:
            y_new.append(0)
    return np.array(y_new)

def max_model(train_x, train_y, learning_rate, epochs, batch_size):
    # define hyper-parameters to be optimised
    verbose = 1
    loss = 'mse'
    optimiser = adam(learning_rate=learning_rate, clipnorm=0.5)
    activation = 'relu'
    # callback = EarlyStopping(monitor='val_loss', patience=10)
    filepath = "LSTM Models/Post Processing/batch_size" + str(batch_size) + "-{epoch:02d}.hdf5"
    callback_model = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False,
                                     save_weights_only=False, mode='auto', period=10)
    # callback_nan = TerminateOnNaN()

    # define model
    inputs = Input(shape=(train_x.shape[1],))
    # dropout = Dropout(0.2)(inputs)
    hidden_vector = Dense(16, activation=activation)(inputs)
    # hidden_vector = Dropout(0.2)(hidden_vector)
    # hidden_vector = Dense(16, activation=activation)(hidden_vector)
    outputs = Dense(1)(hidden_vector)

    model = Model(inputs=inputs, outputs=outputs, name="model")
    print(model.summary())
    model.compile(loss=loss, optimizer=optimiser)
    # fit network
    history = model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose, callbacks=[callback_model])
    return model

def max_thresh_model(train_x, train_y, learning_rate, epochs, batch_size, sample_weights):
    # define hyper-parameters to be optimised
    verbose = 1
    loss = 'binary_crossentropy'
    optimiser = adam(learning_rate=learning_rate, clipnorm=0.5)
    activation = 'relu'
    # callback = EarlyStopping(monitor='val_loss', patience=10)
    filepath = "LSTM Models/Post Processing/batch_size" + str(batch_size) + "-{epoch:02d}.hdf5"
    callback_model = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False,
                                     save_weights_only=False, mode='auto', period=10)
    # callback_nan = TerminateOnNaN()

    # define model
    inputs = Input(shape=(train_x.shape[1],))
    hidden_vector = Dense(5, activation=activation)(inputs)
    outputs = Dense(train_y.shape[1], activation='softmax')(hidden_vector)

    model = Model(inputs=inputs, outputs=outputs, name="model")
    print(model.summary())
    model.compile(loss=loss, optimizer=optimiser, metrics=["accuracy"])
    # fit network
    history = model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose, callbacks=[callback_model], sample_weight=np.array(sample_weights))
    return model

def test_thresh_error(model, test_x, test_y, num_cows=197):
    y_pred = model.predict(test_x)
    print(y_pred)

    samples_per_cow = int(test_x.shape[0]/num_cows)

    # error calc
    false_pos = 0
    false_neg = 0
    total_pos = 0
    total_neg = 0

    # for i in iter:
    for sample_n in range(0, samples_per_cow):
        for cow_n in range(0,num_cows):
            # extract prediction
            i = sample_n + cow_n*samples_per_cow

            max_value = max(y_pred[i])
            max_pred_index = list(y_pred[i]).index(max_value)

            max_value = max(test_y[i])
            max_actual_index = list(test_y[i]).index(max_value)

            # calculate false positive and false negative above and below threshold
            if max_actual_index == 0:
                total_pos += 1
                if max_pred_index == 1:
                    # plot = True
                    false_neg += 1
            else:
                total_neg += 1
                if max_pred_index == 0:
                    # plot = True
                    false_pos += 1

    sensitivity_max_total = (total_pos - false_neg) / total_pos
    specificity_max_total = (total_neg - false_pos) / total_neg

    print("\n")
    print("\n")
    print("\nSENSITIVITY MAX")
    print(sensitivity_max_total)
    print("\nSPECIFICITY MAX")
    print(specificity_max_total)
    print("\n")
    print("\n")

    return sensitivity_max_total, specificity_max_total

def test_max_error(y_pred, test_y, norm_y, thresh, num_cows=197):
    y_pred = norm_y.inverse_transform(y_pred)
    test_y = norm_y.inverse_transform(test_y)

    samples_per_cow = int(test_y.shape[0]/num_cows)
    print(samples_per_cow)

    # error calc
    false_pos = 0
    false_neg = 0
    total_pos = 0
    total_neg = 0
    daily_errors = []

    # for i in iter:
    for sample_n in range(0, samples_per_cow):
        for cow_n in range(0,num_cows):
            # extract prediction
            i = sample_n + cow_n*samples_per_cow

            predicted = y_pred[i][0]
            actual = test_y[i][0]

            error = mean_squared_error([actual], [predicted], squared=False)
            daily_errors.append(error)

            # calculate false positive and false negative above and below threshold
            if actual > thresh:
                total_pos += 1
                if predicted < thresh:
                    # plot = True
                    false_neg += 1
            else:
                total_neg += 1
                if predicted > thresh:
                    # plot = True
                    false_pos += 1

            if False:
                plt.figure()
                print(error)
                plt.plot(y_pred_orig, label='forecast')
                plt.plot(y_actual_orig, label='actual')
                plt.legend()
                plt.show()

    mean_max_error = np.mean(daily_errors)
    sensitivity_max_total = (total_pos - false_neg) / total_pos
    specificity_max_total = (total_neg - false_pos) / total_neg

    print("\n")
    print("\n")
    print("\nRMSE ERROR MAX")
    print(mean_max_error)
    print("\nSENSITIVITY MAX")
    print(sensitivity_max_total)
    print("\nSPECIFICITY MAX")
    print(specificity_max_total)
    print("\n")
    print("\n")

    return mean_max_error, sensitivity_max_total, specificity_max_total

def test_error(y_pred, test_x, test_y, norm_y, num_cows=197,y_prev=0, plot=False):
    test_x = test_x[0]

    samples_per_cow = int(test_x.shape[0]/num_cows)

    # error calc
    false_pos_freq = 0
    false_neg_freq = 0
    total_pos_freq = 0
    total_neg_freq = 0
    false_pos_max = 0
    false_neg_max = 0
    total_pos_max = 0
    total_neg_max = 0
    hourly_errors = []
    daily_errors = []
    max_errors = []
    thresh_count_errors = []

    # for i in iter:
    for sample_n in range(y_prev, samples_per_cow):
        for cow_n in range(0,num_cows):
            # extract prediction
            i = sample_n + cow_n*samples_per_cow
            y_pred_orig = norm_y.inverse_transform(y_pred[i])
            test_y_i = test_y[i].reshape(-1,1)
            y_actual_orig = norm_y.inverse_transform(test_y_i)

            # calculate hourly RMSE
            error = mean_squared_error(y_actual_orig, y_pred_orig, squared=False)

            # calculate daily frequency and max
            freq_actual = sum(y_actual_orig)
            freq_forecast = sum(y_pred_orig)
            max_actual = max(y_actual_orig)
            max_forecast = max(y_pred_orig)

            # error in frequency and max prediction
            daily_error = abs(freq_actual - freq_forecast)
            max_error = abs(max_actual - max_forecast)

            # append errors
            hourly_errors.append(error)
            daily_errors.append(daily_error)
            max_errors.append(max_error)

            # calculate false positive and false negative above and below daily threshold
            if max_actual > 12:
                total_pos_max += 1
                if max_forecast <= 12:
                    # plot = True
                    false_neg_max += 1
            else:
                total_neg_max += 1
                if max_forecast > 12:
                    # plot = True
                    false_pos_max += 1

            # calculate number of hours above a panting threshold
            pred_count = 0
            for hour in y_pred_orig:
                if hour>12:
                    pred_count+=1
            act_count = 0
            for hour in y_actual_orig:
                if hour>12:
                    act_count+=1
            thresh_count_errors.append(abs(act_count-pred_count))

            # calculate false positive and false negative above and below daily threshold
            if act_count > 4:
                total_pos_freq += 1
                if pred_count <= 4:
                    # plot = True
                    false_neg_freq += 1
            else:
                total_neg_freq += 1
                if pred_count > 4:
                    # plot = True
                    false_pos_freq += 1

            if plot:
                plt.figure()
                print(error)
                print(daily_error)
                plt.plot(y_pred_orig, label='forecast')
                plt.plot(y_actual_orig, label='actual')
                plt.legend()
                plt.show()

    # calculate summary errors
    mean_hourly_RMSE = np.mean(hourly_errors)
    RMSE_daily_freq = np.mean(daily_errors)
    RMSE_max = np.mean(max_errors)
    sensitivity_freq_total = (total_pos_freq-false_neg_freq)/total_pos_freq
    specificity_freq_total = (total_neg_freq-false_pos_freq)/total_neg_freq
    sensitivity_max_total = (total_pos_max-false_neg_max)/total_pos_max
    specificity_max_total = (total_neg_max-false_pos_max)/total_neg_max
    thresh_count_RMSE = np.mean(thresh_count_errors)

    print(total_pos_freq)
    print(total_neg_freq)

    print("\n")
    print("\n")
    print("\nMEAN HOURLY RMSE")
    print(mean_hourly_RMSE)
    print("\nRMSE DAILY FREQ")
    print(RMSE_daily_freq)
    print("\nRMSE MAX ERROR")
    print(RMSE_max)
    print("\nSENSITIVITY THRESH")
    print(sensitivity_freq_total)
    print("\nSPECIFICITY THRESH")
    print(specificity_freq_total)
    print("\nSENSITIVITY MAX")
    print(sensitivity_max_total)
    print("\nSPECIFICITY MAX")
    print(specificity_max_total)
    print("\nTHRESH COUNT ERROR")
    print(thresh_count_RMSE)
    print("\n")
    print("\n")

    return mean_hourly_RMSE, RMSE_daily_freq, thresh_count_RMSE, sensitivity_freq_total, specificity_freq_total, RMSE_max, sensitivity_max_total, specificity_max_total

def add_max_to_train(x):
    x_new = []
    for i in range(len(x)):
        x_new.append(np.append(x[i], max(x[i])))
    return np.array(x_new)

def calc_sample_weights(y_thresh):
    count = 0
    total = 0
    sample_weights = []
    for sample in y_thresh:
        total+=1
        if sample[0] == 1:
            count+=1

    # calculate sample weights
    for sample in y_thresh:
        if sample[0] == 1:
            w = 0.5/(count/total)
            w = pow(w,2)
            sample_weights.append(w)
        else:
            w = 0.5/((total-count)/total)
            w = pow(w,2)
            sample_weights.append(w)
    return sample_weights

def calc_herd_data(x, y, num_cows = 197):
    # get samples per cow
    samples_per_cow = int(x.shape[0] / num_cows)

    x_new = []
    y_new = []
    for sample_n in range(0, samples_per_cow):
        x_temp = []
        y_temp = []
        for cow_n in range(0, num_cows):
            # extract prediction
            i = sample_n + cow_n * samples_per_cow
            x_temp.append(x[i])
            y_temp.append(y[i])
        # print(x_temp)
        # print(np.mean(x_temp,axis=0))
        x_new.append(np.mean(x_temp, axis=0))
        y_new.append(np.mean(y_temp, axis=0))

    return np.array(x_new), np.array(y_new)

def test_error_herd(y_pred, test_x, test_y, norm_y, num_cows=197,y_prev=0, plot=False):
    test_x = test_x[0]

    samples_per_cow = int(test_x.shape[0]/num_cows)

    # error calc
    false_pos_freq = 0
    false_neg_freq = 0
    total_pos_freq = 0
    total_neg_freq = 0
    false_pos_max = 0
    false_neg_max = 0
    total_pos_max = 0
    total_neg_max = 0
    hourly_errors = []
    daily_errors = []
    max_errors = []
    thresh_count_errors = []

    # for i in iter:
    for sample_n in range(y_prev, samples_per_cow):
        actual_list = []
        pred_list = []
        for cow_n in range(0,num_cows):
            # extract prediction
            i = sample_n + cow_n*samples_per_cow
            y_pred_orig = norm_y.inverse_transform(y_pred[i])
            test_y_i = test_y[i].reshape(-1,1)
            y_actual_orig = norm_y.inverse_transform(test_y_i)

            # add to list
            actual_list.append(y_actual_orig)
            pred_list.append(y_pred_orig)

        y_actual_herd = np.mean(actual_list, axis=0)
        y_pred_herd = np.mean(pred_list, axis=0)

        # calculate hourly RMSE
        error = mean_squared_error(y_actual_herd, y_pred_herd, squared=False)

        # calculate daily frequency and max
        freq_actual = sum(y_actual_herd)
        freq_forecast = sum(y_pred_herd)
        max_actual = max(y_actual_herd)
        max_forecast = max(y_pred_herd)

        # error in frequency and max prediction
        daily_error = abs(freq_actual - freq_forecast)
        max_error = abs(max_actual - max_forecast)

        # append errors
        hourly_errors.append(error)
        daily_errors.append(daily_error)
        max_errors.append(max_error)

        # calculate false positive and false negative above and below daily threshold
        # if freq_actual > 119:
        #     total_pos_freq += 1
        #     if freq_forecast < 119:
        #         # plot = True
        #         false_neg_freq += 1
        # else:
        #     total_neg_freq += 1
        #     if freq_forecast > 120:
        #         # plot = True
        #         false_pos_freq += 1

        # calculate false positive and false negative above and below daily threshold
        if max_actual > 10:
            total_pos_max += 1
            if max_forecast < 10:
                # plot = True
                false_neg_max += 1
        else:
            total_neg_max += 1
            if max_forecast > 10:
                # plot = True
                false_pos_max += 1

        # calculate number of hours above a panting threshold
        pred_count = 0
        for hour in y_pred_orig:
            if hour > 8:
                pred_count += 1
        act_count = 0
        for hour in y_actual_orig:
            if hour > 8:
                act_count += 1
        thresh_count_errors.append(abs(act_count - pred_count))
        # print("thresh count")
        # print(act_count)
        # print(pred_count)

        # calculate false positive and false negative above and below daily threshold
        if act_count > 4:
            total_pos_freq += 1
            if pred_count <= 4:
                # plot = True
                false_neg_freq += 1
        else:
            total_neg_freq += 1
            if pred_count > 4:
                # plot = True
                false_pos_freq += 1

        if plot:
            plt.figure()
            print(error)
            print(daily_error)
            plt.plot(y_actual_herd, label='actual')
            plt.plot(y_pred_herd, label='forecast')
            plt.legend()
            plt.show()

    # calculate summary errors
    mean_hourly_RMSE = np.mean(hourly_errors)
    RMSE_daily_freq = np.mean(daily_errors)
    RMSE_max = np.mean(max_errors)
    sensitivity_freq_total = (total_pos_freq - false_neg_freq) / total_pos_freq
    specificity_freq_total = (total_neg_freq - false_pos_freq) / total_neg_freq
    sensitivity_max_total = (total_pos_max - false_neg_max) / total_pos_max
    specificity_max_total = (total_neg_max - false_pos_max) / total_neg_max
    thresh_count_RMSE = np.mean(thresh_count_errors)

    print(total_pos_freq)
    print(total_neg_freq)

    print("\n")
    print("\n")
    print("\nMEAN HOURLY RMSE")
    print(mean_hourly_RMSE)
    print("\nRMSE DAILY FREQ")
    print(RMSE_daily_freq)
    print("\nRMSE MAX ERROR")
    print(RMSE_max)
    print("\nSENSITIVITY DAILY FREQ")
    print(sensitivity_freq_total)
    print("\nSPECIFICITY DAILY FREQ")
    print(specificity_freq_total)
    print("\nSENSITIVITY MAX")
    print(sensitivity_max_total)
    print("\nSPECIFICITY MAX")
    print(specificity_max_total)
    print("\nTHRESH COUNT ERROR")
    print(thresh_count_RMSE)
    print("\n")
    print("\n")

    return mean_hourly_RMSE, RMSE_daily_freq, thresh_count_RMSE, sensitivity_freq_total, specificity_freq_total, RMSE_max, sensitivity_max_total, specificity_max_total

def max_threshold_model(x_train_post, y_train_post, x_test_post, y_test_post, forecast_model, scalar_y, learning_rate, batch_size, epochs, herd = False, thresh=12, test=True):
    train_pred = forecast_model.predict(x_train_post)
    test_pred = forecast_model.predict(x_test_post)

    # if herd, take the average
    if herd:
        test_error_herd(test_pred, x_test_post, y_test_post, scalar_y)
        train_pred, y_train_post = calc_herd_data(train_pred, y_train_post)
        test_pred, y_test_post = calc_herd_data(test_pred, y_test_post)
    else:
        if test:
            mean_hourly_RMSE, RMSE_daily_freq, thresh_count_RMSE, sensitivity_freq_total, specificity_freq_total, RMSE_max, sensitivity_max_total, specificity_max_total = test_error(test_pred, x_test_post, y_test_post, scalar_y)
        else:
            sensitivity_max_total = None
            specificity_max_total = None

    # now we can reshape for new model
    test_pred = test_pred.reshape(test_pred.shape[0], test_pred.shape[1])
    train_pred = train_pred.reshape(train_pred.shape[0], train_pred.shape[1])

    # calculate thresholds
    train_thresh = calculate_threshold_y(y_train_post, scalar_y, thresh)
    test_thresh = calculate_threshold_y(y_test_post, scalar_y, thresh)

    sample_weights = calc_sample_weights(train_thresh)

    model = max_thresh_model(train_pred, train_thresh, learning_rate=learning_rate, batch_size=batch_size, epochs=epochs, sample_weights=sample_weights)

    sensitivity_max_new, specificity_max_new = test_thresh_error(model, test_pred, test_thresh, thresh)

    return sensitivity_max_total, specificity_max_total, sensitivity_max_new, specificity_max_new

def max_prediction_model(x_train_post, y_train_post, x_test_post, y_test_post, forecast_model, scalar_y, learning_rate, batch_size, epochs, herd=False, thresh=12):
    train_pred = forecast_model.predict(x_train_post)
    test_pred = forecast_model.predict(x_test_post)

    # if herd, take the average
    if herd:
        test_error_herd(test_pred, x_test_post, y_test_post, scalar_y)
        train_pred, y_train_post = calc_herd_data(train_pred, y_train_post)
        test_pred, y_test_post = calc_herd_data(test_pred, y_test_post)
    else:
        test_error(test_pred, x_test_post, y_test_post, scalar_y)

    # now we can reshape for new model
    train_pred = train_pred.reshape(train_pred.shape[0], train_pred.shape[1])
    test_pred = test_pred.reshape(test_pred.shape[0], test_pred.shape[1])

    # convert to max
    train_max = add_max_y(y_train_post)
    test_max = add_max_y(y_test_post)

    model = max_model(train_pred, train_max, learning_rate=learning_rate, batch_size=batch_size, epochs=epochs)

    y_pred = model.predict(test_pred)

    test_max_error(y_pred, test_max, scalar_y, thresh=thresh)

def weight_regression_samples(train_max_pred, train_max, scalar_y, thresh):
    norm_thresh = scalar_y.transform([[thresh]])
    train_max_pred_new = []
    train_max_new = []
    # include only positive samples
    for i in range(len(train_max)):
        if train_max[i][0]>norm_thresh:
            train_max_new.append(train_max[i])
            train_max_pred_new.append(train_max_pred[i])
        # train_max_new.append(train_max[i])
        # train_max_pred_new.append(train_max_pred[i])

    return train_max_pred_new, train_max_new

def herd_max_regression(forecast_model, x_train_post, y_train_post, x_test_post, y_test_post, scalar_y, plot=False):
    train_pred = forecast_model.predict(x_train_post)
    test_pred = forecast_model.predict(x_test_post)

    # test_error_herd(test_pred, x_test_post, y_test_post, scalar_y)
    train_pred, y_train_post = calc_herd_data(train_pred, y_train_post)
    test_pred, y_test_post = calc_herd_data(test_pred, y_test_post)

    # now we can reshape for new model
    train_pred = train_pred.reshape(train_pred.shape[0], train_pred.shape[1])
    test_pred = test_pred.reshape(test_pred.shape[0], test_pred.shape[1])

    train_max = add_max_y(y_train_post)
    test_max = add_max_y(y_test_post)
    train_max_pred = add_max_y(train_pred)
    test_max_pred = add_max_y(test_pred)

    # weight samples to include positives twice
    # train_max_pred, train_max = weight_regression_samples(train_max_pred, train_max, scalar_y, 10)

    reg = LinearRegression().fit(train_max_pred, train_max)
    new_test_pred = reg.predict(test_max_pred)
    print(reg.coef_)
    print(reg.intercept_)

    if plot:
        plt.title("Post Processing Training Herd Max vs Predicted")
        plt.plot(train_max, label='actual')
        plt.plot(train_max_pred, label='pred')
        plt.ylabel("Normalised Max Herd")
        plt.xlabel("Time Step")
        plt.legend()
        plt.show()

        plt.title("Post Processing Testing Herd Max vs Predicted")
        plt.plot(test_max, label='actual')
        plt.plot(test_max_pred, label='pred')
        plt.plot(new_test_pred, label='adjusted pred')
        plt.ylabel("Normalised Max Herd")
        plt.xlabel("Time Step")
        plt.legend()
        plt.show()

    test_max_error(new_test_pred, test_max, scalar_y, thresh=10, num_cows=1)
    test_max_error(test_max_pred, test_max, scalar_y, thresh=10, num_cows=1)

def test_svm_error(pred, actual):
    # error calc
    false_pos = 0
    false_neg = 0
    total_pos = 0
    total_neg = 0

    for i in range(len(pred)):
        # calculate false positive and false negative above and below threshold
        if actual[i] == 1:
            total_pos += 1
            if pred[i] == 0:
                false_neg += 1
        else:
            total_neg += 1
            if pred[i] == 1:
                false_pos += 1

    sensitivity_max_total = (total_pos - false_neg) / total_pos
    specificity_max_total = (total_neg - false_pos) / total_neg

    print("\n")
    print("\n")
    print("\nSENSITIVITY MAX")
    print(sensitivity_max_total)
    print("\nSPECIFICITY MAX")
    print(specificity_max_total)
    print("\n")
    print("\n")

    return sensitivity_max_total, specificity_max_total

def herd_max_svm(forecast_model, x_train_post, y_train_post, x_test_post, y_test_post, scalar_y, thresh):
    train_pred = forecast_model.predict(x_train_post)
    test_pred = forecast_model.predict(x_test_post)

    # test_error_herd(test_pred, x_test_post, y_test_post, scalar_y)
    train_pred, y_train_post = calc_herd_data(train_pred, y_train_post)
    test_pred, y_test_post = calc_herd_data(test_pred, y_test_post)

    # now we can reshape for new model
    train_pred = train_pred.reshape(train_pred.shape[0], train_pred.shape[1])
    test_pred = test_pred.reshape(test_pred.shape[0], test_pred.shape[1])

    train_thresh = calculate_threshold_svm(y_train_post, scalar_y, thresh)
    test_thresh = calculate_threshold_svm(y_test_post, scalar_y, thresh)
    orig_pred = calculate_threshold_svm(test_pred, scalar_y, thresh)

    train_max_pred = add_max_y(train_pred)
    test_max_pred = add_max_y(test_pred)

    # weight samples to include positives twice
    # train_max_pred, train_max = weight_regression_samples(train_max_pred, train_max, scalar_y, 10)

    classifier = svm.SVC()
    classifier.fit(train_max_pred, train_thresh)
    new_pred = classifier.predict(test_max_pred)

    # print(new_pred)
    # print(orig_pred)
    # print(test_thresh)

    test_svm_error(new_pred, test_thresh)
    test_svm_error(orig_pred, test_thresh)

def run_individual_post_process():
    error_list = []
    # run multivariate
    for batch_size in [64, 128, 256, 512]:
        model_name = 'LSTM Models/Multivariate Optimisation/batch_size' + str(batch_size) + '-100.hdf5'
        forecast_model, x_train, y_train, x_test, y_test, scalar_y = import_model('Deep Learning Data/Multivariate Lag 120', model_name)
        x_train_new, y_train_new, x_test_new, y_test_new = new_test_train(x_test, y_test)
        sens_new_list = []
        spec_new_list = []
        test = True
        for i in range(3):
            sens_old, spec_old, sens_new, spec_new = max_threshold_model(x_train_new, y_train_new, x_test_new, y_test_new, forecast_model, scalar_y,
                                                                         learning_rate=0.001, batch_size=8, epochs=5, herd=False, thresh=12, test=test)
            sens_old_final = sens_old
            spec_old_final = spec_old
            # append test to list
            sens_new_list.append(sens_new)
            spec_new_list.append(spec_new)
            test=False
        error_list.append(['Mulitvariate', batch_size, sens_old_final, spec_old_final, np.mean(sens_new_list), np.mean(spec_new_list)])

    # run univariate
    for batch_size in [64, 128, 256, 512]:
        model_name = 'LSTM Models/Univariate Optimisation/batch_size' + str(batch_size) + '-50.hdf5'
        forecast_model, x_train, y_train, x_test, y_test, scalar_y = import_model('Deep Learning Data/Univariate Lag 120', model_name)
        x_train_new, y_train_new, x_test_new, y_test_new = new_test_train(x_test, y_test)
        sens_new_list = []
        spec_new_list = []
        test = True
        for i in range(3):
            sens_old, spec_old, sens_new, spec_new = max_threshold_model(x_train_new, y_train_new, x_test_new, y_test_new, forecast_model, scalar_y,
                                                                         learning_rate=0.001, batch_size=8, epochs=5, herd=False, thresh=12, test=test)
            sens_old_final = sens_old
            spec_old_final = spec_old
            # append test to list
            sens_new_list.append(sens_new)
            spec_new_list.append(spec_new)
            test=False
        error_list.append(['Mulitvariate', batch_size, np.mean(sens_old_final), spec_old_final, np.mean(sens_new_list), np.mean(spec_new_list)])

    error_summary_df = pd.DataFrame(error_list, columns=['model', 'batch_size', 'old sensitivity', 'old specificity',
                                                         'new_sensitivity', 'new specificity'])
    print(error_summary_df)
    error_summary_df.to_pickle('Post Processing/individual_error_summary_2.pkl')


# # name model
# model_name = 'LSTM Models/Multivariate Optimisation/batch_size256-100.hdf5'
#
# # extract model data
# forecast_model, x_train, y_train, x_test, y_test, scalar_y = import_model('Deep Learning Data/Multivariate Lag 120', model_name)
#
# x_train_new, y_train_new, x_test_new, y_test_new = new_test_train(x_test, y_test)
#
# # herd_max_regression(forecast_model, x_train, y_train, x_test, y_test, scalar_y, True)
# herd_max_svm(forecast_model, x_train, y_train, x_test, y_test, scalar_y, 10)
#
# # max_prediction_model(x_train_new, y_train_new, x_test_new, y_test_new, forecast_model, scalar_y, learning_rate=0.001, batch_size=8, epochs=10, herd=False, thresh=12)
# # max_threshold_model(x_train_new, y_train_new, x_test_new, y_test_new, forecast_model, scalar_y, learning_rate=0.001, batch_size=8, epochs=10, herd=False, thresh=12)

run_individual_post_process()

# small_test_error = pd.read_pickle('Post Processing/individual_error_summary.pkl')
# print(small_test_error.to_string())

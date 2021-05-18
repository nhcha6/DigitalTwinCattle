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

def calculate_threshold_count(y, scalar_y, thresh_pant, thresh_count):
    y_new = []
    y = scalar_y.inverse_transform(y)
    for y_sample in y:
        # calculate number of hours above a panting threshold
        count = 0
        for hour in y_sample:
            if hour > thresh_pant:
                count += 1
        if count>thresh_count:
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

def test_post_thresh_error(model, test_x, test_y, num_cows=197):
    y_pred = model.predict(test_x)

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

def test_post_herd_error(model, test_x, test_y, num_cows=197):
    y_pred = model.predict(test_x)

    samples_per_cow = int(test_x.shape[0] / num_cows)

    # error calc
    false_pos = 0
    false_neg = 0
    total_pos = 0
    total_neg = 0

    # list of percentage positives
    percent_post_actual_list = []
    percent_post_pred_list = []
    # threshold set at 50% positive
    thresh = 0.5
    for sample_n in range(0, samples_per_cow):
        actual_pos = 0
        predicted_pos = 0
        for cow_n in range(0, num_cows):
            # extract prediction
            i = sample_n + cow_n * samples_per_cow

            max_value = max(y_pred[i])
            max_pred_index = list(y_pred[i]).index(max_value)

            max_value = max(test_y[i])
            max_actual_index = list(test_y[i]).index(max_value)

            # calculate false positive and false negative above and below threshold
            if max_actual_index == 0:
                actual_pos += 1
            if max_pred_index == 0:
                predicted_pos += 1

        percent_pos_actual = actual_pos/197
        percent_pos_predicted = predicted_pos/197

        percent_post_actual_list.append(percent_pos_actual)
        percent_post_pred_list.append(percent_pos_predicted)

        if percent_pos_actual > thresh:
            total_pos += 1
            if percent_pos_predicted < thresh:
                # plot = True
                false_neg += 1
        else:
            total_neg += 1
            if percent_pos_predicted > thresh:
                # plot = True
                false_pos += 1

    pos_percent_error = mean_squared_error(percent_post_actual_list, percent_post_pred_list, squared=False)
    # print(pos_percent_error)
    # print(total_pos/(total_pos+total_neg))

    sensitivity_max_total = (total_pos - false_neg) / total_pos
    specificity_max_total = (total_neg - false_pos) / total_neg

    print("\n")
    print("\n")
    print("\nSENSITIVITY HERD MAX")
    print(sensitivity_max_total)
    print("\nSPECIFICITY HERD MAX")
    print(specificity_max_total)
    print("\n")
    print("\n")


def test_threshold_errors(y_pred, test_x, test_y, norm_y, thresh_pant, thresh_count, num_cows=197):
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
    for sample_n in range(0, samples_per_cow):
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
            if max_actual > thresh_pant:
                total_pos_max += 1
                if max_forecast <= thresh_pant:
                    # plot = True
                    false_neg_max += 1
            else:
                total_neg_max += 1
                if max_forecast > thresh_pant:
                    # plot = True
                    false_pos_max += 1

            # calculate number of hours above a panting threshold
            pred_count = 0
            for hour in y_pred_orig:
                if hour>thresh_pant:
                    pred_count+=1
            act_count = 0
            for hour in y_actual_orig:
                if hour>thresh_pant:
                    act_count+=1
            thresh_count_errors.append(abs(act_count-pred_count))

            # calculate false positive and false negative above and below daily threshold
            if act_count > thresh_count:
                total_pos_freq += 1
                if pred_count <= thresh_count:
                    # plot = True
                    false_neg_freq += 1
            else:
                total_neg_freq += 1
                if pred_count > thresh_count:
                    # plot = True
                    false_pos_freq += 1

    pos_percentage_count = total_pos_freq/(total_pos_freq+total_neg_freq)
    pos_percentage_max = total_pos_max/(total_pos_max+total_neg_max)
    # calculate summary errors
    mean_hourly_RMSE = np.mean(hourly_errors)
    RMSE_daily_freq = np.mean(daily_errors)
    RMSE_max = np.mean(max_errors)
    sensitivity_freq_total = (total_pos_freq-false_neg_freq)/total_pos_freq
    specificity_freq_total = (total_neg_freq-false_pos_freq)/total_neg_freq
    sensitivity_max_total = (total_pos_max-false_neg_max)/total_pos_max
    specificity_max_total = (total_neg_max-false_pos_max)/total_neg_max
    thresh_count_RMSE = np.mean(thresh_count_errors)

    print("\n")
    print("\n")
    print("\n% POS MAX")
    print(pos_percentage_max)
    print("\n% POS COUNT")
    print(pos_percentage_count)
    print("\nSENSITIVITY THRESH")
    print(sensitivity_freq_total)
    print("\nSPECIFICITY THRESH")
    print(specificity_freq_total)
    print("\nSENSITIVITY MAX")
    print(sensitivity_max_total)
    print("\nSPECIFICITY MAX")
    print(specificity_max_total)
    print("\n")
    print("\n")

    return sensitivity_freq_total, specificity_freq_total, sensitivity_max_total, specificity_max_total, pos_percentage_max, pos_percentage_count

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

def max_threshold_model(x_train_post, y_train_post, x_test_post, y_test_post, forecast_model, scalar_y, learning_rate, batch_size, epochs, thresh=12, test=True):
    train_pred = forecast_model.predict(x_train_post)
    test_pred = forecast_model.predict(x_test_post)

    # if herd, take the average
    if test:
        sensitivity_freq_total, specificity_freq_total, sensitivity_max_total, specificity_max_total, pos_percentage_max, pos_percentage_count = test_threshold_errors(test_pred, x_test_post, y_test_post, scalar_y, thresh, 4)
    else:
        sensitivity_max_total = None
        specificity_max_total = None
        pos_percentage_max = None
        pos_percentage_count = None

    # now we can reshape for new model
    test_pred = test_pred.reshape(test_pred.shape[0], test_pred.shape[1])
    train_pred = train_pred.reshape(train_pred.shape[0], train_pred.shape[1])

    # calculate thresholds
    train_thresh = calculate_threshold_y(y_train_post, scalar_y, thresh)
    test_thresh = calculate_threshold_y(y_test_post, scalar_y, thresh)

    sample_weights = calc_sample_weights(train_thresh)

    # print(train_thresh)
    # print(sample_weights)

    model = max_thresh_model(train_pred, train_thresh, learning_rate=learning_rate, batch_size=batch_size, epochs=epochs, sample_weights=sample_weights)

    sensitivity_max_new, specificity_max_new = test_post_thresh_error(model, test_pred, test_thresh)

    return sensitivity_max_total, specificity_max_total, sensitivity_max_new, specificity_max_new, pos_percentage_max, pos_percentage_count

def max_count_threshold_model(x_train_post, y_train_post, x_test_post, y_test_post, forecast_model, scalar_y, learning_rate, batch_size, epochs, thresh_pant=9, thresh_count = 4, test=True):
    train_pred = forecast_model.predict(x_train_post)
    test_pred = forecast_model.predict(x_test_post)

    # if herd, take the average
    if test:
        sensitivity_freq_total, specificity_freq_total, sensitivity_max_total, specificity_max_total, pos_percentage_max, pos_percentage_count = test_threshold_errors(test_pred, x_test_post, y_test_post, scalar_y, thresh_pant, thresh_count)
    else:
        sensitivity_freq_total = None
        specificity_freq_total = None
        pos_percentage_max = None
        pos_percentage_count = None

    # now we can reshape for new model
    test_pred = test_pred.reshape(test_pred.shape[0], test_pred.shape[1])
    train_pred = train_pred.reshape(train_pred.shape[0], train_pred.shape[1])

    # calculate thresholds
    train_thresh = calculate_threshold_count(y_train_post, scalar_y, thresh_pant, thresh_count)
    test_thresh = calculate_threshold_count(y_test_post, scalar_y, thresh_pant, thresh_count)

    sample_weights = calc_sample_weights(train_thresh)

    model = max_thresh_model(train_pred, train_thresh, learning_rate=learning_rate, batch_size=batch_size, epochs=epochs, sample_weights=sample_weights)

    sensitivity_max_new, specificity_max_new = test_post_thresh_error(model, test_pred, test_thresh)

    # test_post_herd_error(model, test_pred, test_thresh)

    return sensitivity_freq_total, specificity_freq_total, sensitivity_max_new, specificity_max_new, pos_percentage_max, pos_percentage_count

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
    pos_percentage = total_pos/(total_neg+total_pos)


    print("\n")
    print("\n")
    print("\nTOTAL POSITIVE")
    print(pos_percentage)
    print("\nSENSITIVITY MAX")
    print(sensitivity_max_total)
    print("\nSPECIFICITY MAX")
    print(specificity_max_total)
    print("\n")
    print("\n")

    return sensitivity_max_total, specificity_max_total, pos_percentage

def herd_max_svm(forecast_model, x_train_post, y_train_post, x_test_post, y_test_post, scalar_y, thresh, a=0):
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

    sensitivity_max_old, specificity_max_old, pos_perc = test_svm_error(orig_pred, test_thresh)

    if not a:
        a = pow(2,5*(sensitivity_max_old - specificity_max_old))
        print(a)

    classifier = svm.SVC(class_weight={0:a, 1:1})
    classifier.fit(train_max_pred, train_thresh)
    new_pred = classifier.predict(test_max_pred)

    sensitivity_max_new, specificity_max_new, pos_perc = test_svm_error(new_pred, test_thresh)


    return sensitivity_max_old, specificity_max_old, sensitivity_max_new, specificity_max_new, pos_perc

def vary_thresh_test(model_name):
    model, x_train, y_train, x_test, y_test, scalar_y = import_model('Deep Learning Data/Multivariate Lag 120', model_name)
    test_pred = model.predict(x_test)
    errors = []
    # run pant thresh and count thresh grid search
    for thresh_pant in range(8,21):
        for thresh_count in range(2,9):
            print("THRESH PANT: " + str(thresh_pant))
            print("THRESH COUNT: " + str(thresh_count))
            sensitivity_freq_total, specificity_freq_total, sensitivity_max_total, specificity_max_total, pos_percentage_max, pos_percentage_count = test_threshold_errors(test_pred, x_test, y_test, scalar_y, thresh_pant, thresh_count)
            errors.append([thresh_pant, thresh_count, sensitivity_freq_total, specificity_freq_total, sensitivity_max_total,specificity_max_total, pos_percentage_max, pos_percentage_count])
    # extra points for max thresh analysis
    # for thresh_pant in range(15,17):
    #     thresh_count = 4
    #     print("THRESH PANT: " + str(thresh_pant))
    #     print("THRESH COUNT: " + str(thresh_count))
    #     sensitivity_freq_total, specificity_freq_total, sensitivity_max_total, specificity_max_total, pos_percentage_max, pos_percentage_count = test_threshold_errors(test_pred, x_test, y_test, scalar_y, thresh_pant, thresh_count)
    #     errors.append([thresh_pant, thresh_count, sensitivity_freq_total, specificity_freq_total, sensitivity_max_total, specificity_max_total, pos_percentage_max, pos_percentage_count])
    error_summary_df = pd.DataFrame(errors, columns=['pant thresh', 'count thresh', 'count sensitivity', 'count specificity', 'max sensitivity', 'max specificity', 'perc max', 'perc count'])
    print(error_summary_df.to_string())
    error_summary_df.to_pickle('Post Processing/thresh_test_errors_big.pkl')

def compare_post_process_acc(file_name_1, file_name_2, name_1, name_2):
    errors_df_1 = pd.read_pickle(file_name_1)
    errors_df_2 = pd.read_pickle(file_name_2)
    # add accuracy
    errors_df_1["new accuracy"] = errors_df_1["new_sensitivity"]*errors_df_1["pos"] + errors_df_1["new specificity"]*(1-errors_df_1["pos"])
    errors_df_2["new accuracy"] = errors_df_2["new_sensitivity"]*errors_df_2["pos"] + errors_df_2["new specificity"]*(1-errors_df_2["pos"])

    # multivariate model plots
    multivariate_errors_df_1 = errors_df_1[errors_df_1["model"] == "Univariate"]
    multivariate_errors_df_2 = errors_df_2[errors_df_2["model"] == "Univariate"]

    plt.figure()
    plt.title("Multivariate 100 Epoch Models - Post Process Accuracy vs Batch Size")
    plt.plot(multivariate_errors_df_1["batch_size"], multivariate_errors_df_1["new accuracy"], label=name_1)
    plt.plot(multivariate_errors_df_2["batch_size"], multivariate_errors_df_2["new accuracy"], label=name_2)
    plt.legend()
    plt.xlabel("Batch Size")
    plt.ylabel("Accuracy")
    plt.show()


def plot_post_process_results(file_name, model_name):
    errors_df = pd.read_pickle(file_name)
    print(errors_df.to_string())
    # add accuracy
    errors_df["old accuracy"] = errors_df["old sensitivity"]*errors_df["pos"] + errors_df["old specificity"]*(1-errors_df["pos"])
    errors_df["new accuracy"] = errors_df["new_sensitivity"]*errors_df["pos"] + errors_df["new specificity"]*(1-errors_df["pos"])

    # multivariate model plots
    multivariate_errors_df = errors_df[errors_df["model"]=="Mulitvariate"]

    plt.figure()
    plt.title("Multivariate " + model_name+ " - Sensitivity vs Batch Size")
    plt.plot(multivariate_errors_df["batch_size"], multivariate_errors_df["old sensitivity"], label = "Original")
    plt.plot(multivariate_errors_df["batch_size"], multivariate_errors_df["new_sensitivity"], label = "Post-processing")
    plt.legend()
    plt.xlabel("Batch Size")
    plt.ylabel("Sensitivity")

    plt.figure()
    plt.title("Multivariate " + model_name + " - Specificity vs Batch Size")
    plt.plot(multivariate_errors_df["batch_size"], multivariate_errors_df["old specificity"], label="Original")
    plt.plot(multivariate_errors_df["batch_size"], multivariate_errors_df["new specificity"], label="Post-processing")
    plt.legend()
    plt.xlabel("Batch Size")
    plt.ylabel("Specificity")

    plt.figure()
    plt.title("Multivariate " + model_name + " - Accuracy vs Batch Size")
    plt.plot(multivariate_errors_df["batch_size"], multivariate_errors_df["old accuracy"], label="Original")
    plt.plot(multivariate_errors_df["batch_size"], multivariate_errors_df["new accuracy"], label="Post-processing")
    plt.legend()
    plt.xlabel("Batch Size")
    plt.ylabel("Accuracy")
    plt.show()

    # univariate model plots
    multivariate_errors_df = errors_df[errors_df["model"] == "Univariate"]

    plt.figure()
    plt.title("Univariate " + model_name + " - Sensitivity vs Batch Size")
    plt.plot(multivariate_errors_df["batch_size"], multivariate_errors_df["old sensitivity"], label="Original")
    plt.plot(multivariate_errors_df["batch_size"], multivariate_errors_df["new_sensitivity"], label="Post-processing")
    plt.legend()
    plt.xlabel("Batch Size")
    plt.ylabel("Sensitivity")

    plt.figure()
    plt.title("Univariate " + model_name + " - Specificity vs Batch Size")
    plt.plot(multivariate_errors_df["batch_size"], multivariate_errors_df["old specificity"], label="Original")
    plt.plot(multivariate_errors_df["batch_size"], multivariate_errors_df["new specificity"], label="Post-processing")
    plt.legend()
    plt.xlabel("Batch Size")
    plt.ylabel("Specificity")

    plt.figure()
    plt.title("Univariate " + model_name + " - Accuracy vs Batch Size")
    plt.plot(multivariate_errors_df["batch_size"], multivariate_errors_df["old accuracy"], label="Original")
    plt.plot(multivariate_errors_df["batch_size"], multivariate_errors_df["new accuracy"], label="Post-processing")
    plt.legend()
    plt.xlabel("Batch Size")
    plt.ylabel("Accuracy")
    plt.show()

def plot_thresh_test(file_name):
    thresh_errors_df = pd.read_pickle(file_name)
    # print(thresh_errors_df.to_string())

    # add accuracy
    thresh_errors_df["count accuracy"] = thresh_errors_df["count sensitivity"]*thresh_errors_df["perc count"] + thresh_errors_df["count specificity"]*(1-thresh_errors_df["perc count"])
    thresh_errors_df["max accuracy"] = thresh_errors_df["max sensitivity"]*thresh_errors_df["perc max"] + thresh_errors_df["max specificity"]*(1-thresh_errors_df["perc max"])

    pant_thresh_df = thresh_errors_df.groupby("pant thresh").mean()

    for var in ["sensitivity", "specificity", "accuracy"]:
        plt.figure()
        plt.title(var.title() + ' Against Panting Threshold')
        plt.plot(pant_thresh_df.index, pant_thresh_df["max " + var], label="Max " + var.title())
        plt.plot(pant_thresh_df.index, thresh_errors_df[thresh_errors_df["count thresh"]==8]["count " + var], label="Count Thresh: 8 " + var.title())
        plt.plot(pant_thresh_df.index, thresh_errors_df[thresh_errors_df["count thresh"]==7]["count " + var], label="Count Thresh: 7 " + var.title())
        plt.plot(pant_thresh_df.index, thresh_errors_df[thresh_errors_df["count thresh"]==6]["count " + var], label="Count Thresh: 6 " + var.title())
        plt.plot(pant_thresh_df.index, thresh_errors_df[thresh_errors_df["count thresh"]==5]["count " + var], label="Count Thresh: 5 " + var.title())
        plt.plot(pant_thresh_df.index, thresh_errors_df[thresh_errors_df["count thresh"]==4]["count " + var], label="Count Thresh: 4 " + var.title())
        plt.plot(pant_thresh_df.index, thresh_errors_df[thresh_errors_df["count thresh"]==3]["count " + var], label="Count Thresh: 3 " + var.title())
        plt.plot(pant_thresh_df.index, thresh_errors_df[thresh_errors_df["count thresh"]==2]["count " + var], label="Count Thresh: 2 " + var.title())
        plt.legend()
        plt.xlabel('Panting Threshold')
        plt.ylabel(var.title())

    for var in ["sensitivity", "specificity", "accuracy"]:
        plt.figure()
        plt.title(var.title() + ' Against Percentage Positive')
        plt.plot(pant_thresh_df["perc max"], pant_thresh_df["max " + var], label="Max " + var.title())
        plt.plot(thresh_errors_df[thresh_errors_df["count thresh"]==8]["perc count"], thresh_errors_df[thresh_errors_df["count thresh"]==8]["count " + var], label="Count Thresh: 8 " + var.title())
        plt.plot(thresh_errors_df[thresh_errors_df["count thresh"]==7]["perc count"], thresh_errors_df[thresh_errors_df["count thresh"]==7]["count " + var], label="Count Thresh: 7 " + var.title())
        plt.plot(thresh_errors_df[thresh_errors_df["count thresh"]==6]["perc count"], thresh_errors_df[thresh_errors_df["count thresh"]==6]["count " + var], label="Count Thresh: 6 " + var.title())
        plt.plot(thresh_errors_df[thresh_errors_df["count thresh"]==5]["perc count"], thresh_errors_df[thresh_errors_df["count thresh"]==5]["count " + var], label="Count Thresh: 5 " + var.title())
        plt.plot(thresh_errors_df[thresh_errors_df["count thresh"]==4]["perc count"], thresh_errors_df[thresh_errors_df["count thresh"]==4]["count " + var], label="Count Thresh: 4 " + var.title())
        plt.plot(thresh_errors_df[thresh_errors_df["count thresh"]==3]["perc count"], thresh_errors_df[thresh_errors_df["count thresh"]==3]["count " + var], label="Count Thresh: 3 " + var.title())
        plt.plot(thresh_errors_df[thresh_errors_df["count thresh"]==2]["perc count"], thresh_errors_df[thresh_errors_df["count thresh"]==2]["count " + var], label="Count Thresh: 2 " + var.title())
        plt.legend()
        plt.xlabel('Percentage Postive')
        plt.ylabel(var.title())

    plt.show()

def run_individual_post_process(thresh_pant=12, thresh_count=0):
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
            if thresh_count == 0:
                sens_old, spec_old, sens_new, spec_new, pos_perc_max, pos_perc_count = max_threshold_model(x_train_new, y_train_new, x_test_new, y_test_new, forecast_model, scalar_y,
                                                                         learning_rate=0.001, batch_size=8, epochs=10, thresh=thresh_pant, test=test)
            else:
                sens_old, spec_old, sens_new, spec_new, pos_perc_max, pos_perc_count = max_count_threshold_model(x_train_new, y_train_new, x_test_new,y_test_new, forecast_model, scalar_y,
                                                                             learning_rate=0.001, batch_size=8, epochs=10, thresh_pant=thresh_pant, thresh_count=thresh_count, test=test)
            if test:
                sens_old_final = sens_old
                spec_old_final = spec_old
                if thresh_count == 0:
                    pos_perc = pos_perc_max
                else:
                    pos_perc = pos_perc_count
            # append test to list
            sens_new_list.append(sens_new)
            spec_new_list.append(spec_new)
            test=False
        error_list.append(['Mulitvariate', batch_size, sens_old_final, spec_old_final, np.mean(sens_new_list), np.mean(spec_new_list), pos_perc])

    # run univariate
    for batch_size in [64, 128, 256, 512]:
        model_name = 'LSTM Models/Univariate Optimisation/batch_size' + str(batch_size) + '-50.hdf5'
        forecast_model, x_train, y_train, x_test, y_test, scalar_y = import_model('Deep Learning Data/Univariate Lag 120', model_name)
        x_train_new, y_train_new, x_test_new, y_test_new = new_test_train(x_test, y_test)
        sens_new_list = []
        spec_new_list = []
        test = True
        for i in range(3):
            if thresh_count == 0:
                sens_old, spec_old, sens_new, spec_new, pos_perc_max, pos_perc_count = max_threshold_model(x_train_new, y_train_new, x_test_new, y_test_new, forecast_model, scalar_y,
                                                                         learning_rate=0.001, batch_size=8, epochs=10, thresh=thresh_pant, test=test)
            else:
                sens_old, spec_old, sens_new, spec_new, pos_perc_max, pos_perc_count = max_count_threshold_model(x_train_new, y_train_new, x_test_new,y_test_new, forecast_model, scalar_y,
                                                                             learning_rate=0.001, batch_size=8, epochs=10, thresh_pant=thresh_pant, thresh_count=thresh_count, test=test)
            if test:
                sens_old_final = sens_old
                spec_old_final = spec_old
                if thresh_count == 0:
                    pos_perc = pos_perc_max
                else:
                    pos_perc = pos_perc_count
            # append test to list
            sens_new_list.append(sens_new)
            spec_new_list.append(spec_new)
            test=False
        error_list.append(['Univariate', batch_size, sens_old_final, spec_old_final, np.mean(sens_new_list), np.mean(spec_new_list), pos_perc])

    error_summary_df = pd.DataFrame(error_list, columns=['model', 'batch_size', 'old sensitivity', 'old specificity',
                                                         'new_sensitivity', 'new specificity', 'pos'])
    print(error_summary_df)
    error_summary_df.to_pickle('Post Processing/individual_post_process_error.pkl')

def run_herd_post_process(thresh=10):
    error_list = []
    # run multivariate
    for batch_size in [64, 128, 256, 512]:
        model_name = 'LSTM Models/Multivariate Optimisation/batch_size' + str(batch_size) + '-100.hdf5'
        forecast_model, x_train, y_train, x_test, y_test, scalar_y = import_model('Deep Learning Data/Multivariate Lag 120', model_name)
        sens_old, spec_old, sens_new, spec_new, pos_perc = herd_max_svm(forecast_model, x_train, y_train, x_test, y_test, scalar_y, thresh, a = 1)
        error_list.append(['Mulitvariate', batch_size, sens_old, spec_old, sens_new, spec_new, pos_perc])

    # run univariate
    for batch_size in [64, 128, 256, 512]:
        model_name = 'LSTM Models/Univariate Optimisation/batch_size' + str(batch_size) + '-50.hdf5'
        forecast_model, x_train, y_train, x_test, y_test, scalar_y = import_model('Deep Learning Data/Univariate Lag 120', model_name)
        sens_old, spec_old, sens_new, spec_new, pos_perc = herd_max_svm(forecast_model, x_train, y_train, x_test, y_test, scalar_y, thresh, a = 1)
        error_list.append(['Univariate', batch_size, sens_old, spec_old, sens_new, spec_new, pos_perc])

    error_summary_df = pd.DataFrame(error_list, columns=['model', 'batch_size', 'old sensitivity', 'old specificity', 'new_sensitivity', 'new specificity', 'pos'])
    print(error_summary_df)
    error_summary_df.to_pickle('Post Processing/herd_post_process_error.pkl')

def edit_inputs(x_train, x_test, lag, test_name = 'multivariate'):
    print('Editing Inputs')
    # reduce number of lags
    if lag != 200:
        x_train, x_test = edit_num_lags(x_train, x_test, lag)

    # change to uni or bivariate data
    if test_name[0:10] == 'univariate':
        x_train, x_test = convert_to_univariate(x_train, x_test)
    if test_name[0:9] == 'bivariate':
        x_train, x_test = convert_to_bivariate(x_train, x_test)

    return x_train, x_test

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

def edit_num_lags(train_x, test_x, new_lag):
    train_x[0] = reduce_lags(new_lag, train_x[0])
    test_x[0] = reduce_lags(new_lag, test_x[0])
    return train_x, test_x

def reduce_lags(new_lag, l):
    l = list(l)
    for i in range(len(l)):
        l[i] = l[i][-new_lag:]
    return np.array(l)

################# REPORT CODE #################

# # generate the the errors for different thresholds and save to file
# vary_thresh_test(LSTM Models/Multivariate Optimisation/batch_size256-100.hdf5)
# plot generated error data frame
# plot_thresh_test('Post Processing/thresh_test_errors_big.pkl')
#
# # run individual animal post processing tests tests
# run_individual_post_process(14)
# run_individual_post_process(8, 5)

# # plot the results
# plot_post_process_results("Post Processing/individual_post_process_error_count.pkl", '100 Epoch Models')
# plot_post_process_results("Post Processing/individual_post_process_error_max.pkl", '100 Epoch Models')
# compare_post_process_acc("Post Processing/individual_post_process_error_count.pkl", "Post Processing/individual_post_process_error_max.pkl", "Panting Thresh 8, Count Thresh 5", "Maximum Thresh 12")

# # run herd post process and plot
# run_herd_post_process(11)
# plot_post_process_results("Post Processing/herd_post_process_error_weights.pkl", '100 Epoch Models')

################# ROUGH CODE BELOW #################

# # name model
# model_name = 'LSTM Models/multivariate n_fold/4_fold/4_fold-batch_size128-100.hdf5'
# forecast_model, x_train, y_train, x_test, y_test, scalar_y = import_model('Deep Learning Data/Multivariate Lag 200/4_fold', model_name)
# x_train, x_test = edit_inputs(x_train, x_test, 120, test_name='multivariate')
#
#
# # x_train_new, y_train_new, x_test_new, y_test_new = new_test_train(x_test, y_test)
# x_train_new, y_train_new, x_test_new, y_test_new = [x_train, y_train, x_test, y_test]
# sens_old, spec_old, sens_new, spec_new, pos_perc_max, pos_perc_count = max_count_threshold_model(x_train_new,
#                                                                                                  y_train_new,
#                                                                                                  x_test_new, y_test_new,
#                                                                                                  forecast_model,
#                                                                                                  scalar_y,
#                                                                                                  learning_rate=0.001,
#                                                                                                  batch_size=8,
#                                                                                                  epochs=10,
#                                                                                                  thresh_pant=12,
#                                                                                                  thresh_count=4,
#                                                                                                  test=True)



# vary_thresh_test(LSTM Models/Multivariate Optimisation/batch_size256-100.hdf5)
# plot_thresh_test('Post Processing/thresh_test_errors_big.pkl')

# plot_post_process_results("Post Processing/individual_post_process_error_count.pkl", '100 Epoch Models')
# plot_post_process_results("Post Processing/herd_post_process_error_noweights.pkl", '100 Epoch Models')
# compare_post_process_acc("Post Processing/individual_post_process_error_count.pkl", "Post Processing/individual_post_process_error_max.pkl", "Panting Thresh 8, Count Thresh 5", "Maximum Thresh 12")


# extract model data
# forecast_model, x_train, y_train, x_test, y_test, scalar_y = import_model('Deep Learning Data/Multivariate Lag 120', model_name)
#
# x_train_new, y_train_new, x_test_new, y_test_new = new_test_train(x_test, y_test)
#
# max_threshold_model(x_train_new, y_train_new, x_test_new, y_test_new,forecast_model, scalar_y,learning_rate=0.001, batch_size=8, epochs=5,thresh=12, test=False)
#
# # herd_max_svm(forecast_model, x_train, y_train, x_test, y_test, scalar_y, 13)
#
# herd_max_svm(forecast_model, x_train, y_train, x_test, y_test, scalar_y, 10)

# run_individual_post_process(14)
# run_individual_post_process(8, 5)

# run_herd_post_process(11)

# thresh_errors_df = pd.read_pickle("Post Processing/individual_post_process_error_count.pkl")
# # thresh_errors_df = thresh_errors_df.rename(columns={'pos perc': 'pos'})
# # # thresh_errors_df["pos"] = 0.26507430580149266
# # thresh_errors_df.to_pickle("Post Processing/individual_post_process_error_count.pkl")
#
# print(thresh_errors_df.to_string())
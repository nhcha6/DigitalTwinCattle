import pandas as pd
import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from matplotlib import pyplot as plt
from keras import models
import pickle
from post_processing_model import *

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

def create_category_dict(category_header, cow_ID_header, details_df):
    categories = set(details_df[category_header])
    category_dict = {key: [] for key in categories}
    for category in categories:
        category_dict[category] = details_df[cow_ID_header][details_df[category_header] == category].tolist()
    #print(category_dict)
    return category_dict

def timeseries_error(y_pred, test_x, test_y, train_x, norm_y, num_cows=197,y_prev=0, plot=False):
    forecast = test_x[1]
    test_x = test_x[0]

    samples_per_cow = int(test_x.shape[0]/num_cows)

    # error calc
    timeseries_errors = []
    timeseries_errors_ignore = []
    weather_timeseries = []

    for sample_n in range(y_prev, samples_per_cow):
        sample_errors = []
        sample_errors_ignore = []
        for cow_n in range(0,num_cows):
            # extract prediction
            i = sample_n + cow_n*samples_per_cow
            y_pred_orig = norm_y.inverse_transform(y_pred[i])
            test_y_i = test_y[i].reshape(-1,1)
            y_actual_orig = norm_y.inverse_transform(test_y_i)

            # calculate hourly RMSE
            error = mean_squared_error(y_actual_orig, y_pred_orig, squared=False)
            sample_errors.append(error)

            if sum(y_pred_orig)/24 < 30:
                sample_errors_ignore.append(error)

        if sample_n % 24 == 0:
            weather_timeseries.extend([4*x for x in forecast[i]])
            # print(weather_timeseries)

        # timeseries_errors.append(sum(sample_errors)/len(sample_errors))
        timeseries_errors.append(np.median(sample_errors))
        # timeseries_errors_ignore.append(sum(sample_errors_ignore)/len(sample_errors_ignore))
        timeseries_errors_ignore.append(np.median(sample_errors_ignore))

    return timeseries_errors, timeseries_errors_ignore, weather_timeseries

def calc_cow_errors(y_pred, test_x, test_y, train_x, norm_y, num_cows=197,y_prev=0, plot=False):
    test_x = test_x[0]

    samples_per_cow = int(test_x.shape[0]/num_cows)

    # error calc
    cow_errors = []

    for cow_n in range(0, num_cows):
        cow = []
        for sample_n in range(y_prev, samples_per_cow):
            # extract prediction
            i = sample_n + cow_n*samples_per_cow
            y_pred_orig = norm_y.inverse_transform(y_pred[i])
            test_y_i = test_y[i].reshape(-1,1)
            y_actual_orig = norm_y.inverse_transform(test_y_i)

            # calculate hourly RMSE
            error = mean_squared_error(y_actual_orig, y_pred_orig, squared=False)
            cow.append(error)

        average_error = sum(cow)/len(cow)
        cow_errors.append(average_error)


    return cow_errors

def calc_R_squared(x_values, y_values):
    x_values = np.array(x_values)
    y_values = np.array(y_values)
    x_values = x_values.reshape(-1,)
    y_values = y_values.reshape(-1,)

    correlation_matrix = np.corrcoef(x_values.astype(float), y_values.astype(float))
    correlation_xy = correlation_matrix[0, 1]
    # plt.figure()
    # plt.plot(x_values, y_values, 'bo')
    # plt.show()
    r_squared = correlation_xy ** 2
    # print(r_squared)
    return r_squared

def test_error(y_pred, test_x, test_y, norm_y, num_cows=197,y_prev=0, plot=False, skip_4_fold = False):
    forecast = test_x[1]
    test_x = test_x[0]

    samples_per_cow = int(test_y.shape[0]/num_cows)

    # print(samples_per_cow)

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
    r_squareds = []
    pred = []
    actual = []
    daily_errors = []
    max_errors = []
    thresh_count_errors = []
    inf_count = 0

    # for i in iter:
    samples = 0
    for sample_n in range(y_prev, samples_per_cow):
        # skip the
        # print(skip_4_fold)
        # print(sample_n)
        if (skip_4_fold) and (sample_n in [a for a in range(192, 222)]):
            print('skipped')
            continue
        for cow_n in range(0,num_cows):
            samples += 1
            # extract prediction
            i = sample_n + cow_n*samples_per_cow
            y_pred_orig = norm_y.inverse_transform(y_pred[i])
            test_y_i = test_y[i].reshape(-1,1)
            y_actual_orig = norm_y.inverse_transform(test_y_i)

            # check if the prediction is reasonable: hottest average panting on record is low mid 20s
            if abs(sum(y_pred_orig)/24) > 30:
                inf_count += 1
                print(inf_count)
                print(sum(y_pred_orig)/24)
                continue

            # calculate hourly RMSE
            error = mean_squared_error(y_actual_orig, y_pred_orig, squared=False)
            r_squared = calc_R_squared(y_actual_orig, y_pred_orig)
            pred.extend(y_pred_orig)
            actual.extend(y_actual_orig)

            # if error > 20:
            #     print(sample_n)
            #     print(sum(y_pred_orig)/24)
            #     plt.figure()
            #     plt.plot(y_pred_orig, label='predited')
            #     plt.plot(y_actual_orig, label='actual')
            #     plt.legend()
            #
            #     plt.figure()
            #     plt.plot(test_x[i])
            #
            #     plt.figure()
            #     plt.plot(forecast[i])
            #     plt.show()

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
            r_squareds.append(r_squared)
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

    print("Total Pos: " + str(total_pos_max/samples))

    # calculate summary errors
    mean_hourly_RMSE = np.mean(hourly_errors)
    median_hourly_RMSE = np.median(hourly_errors)
    mean_r_squared = calc_R_squared(actual, pred)
    RMSE_daily_freq = np.mean(daily_errors)
    RMSE_max = np.mean(max_errors)
    sensitivity_freq_total = (total_pos_freq-false_neg_freq)/total_pos_freq
    specificity_freq_total = (total_neg_freq-false_pos_freq)/total_neg_freq
    sensitivity_max_total = (total_pos_max-false_neg_max)/total_pos_max
    specificity_max_total = (total_neg_max-false_pos_max)/total_neg_max
    thresh_count_RMSE = np.mean(thresh_count_errors)

    print("\n")
    print("\n")
    print("\nMEAN HOURLY RMSE")
    print(mean_hourly_RMSE)
    print("\nMEDIAN HOURLY RMSE")
    print(median_hourly_RMSE)
    print("\nMEAN R SQUARED")
    print(mean_r_squared)
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

    return mean_hourly_RMSE, median_hourly_RMSE, mean_r_squared, RMSE_daily_freq, thresh_count_RMSE, sensitivity_freq_total, specificity_freq_total, RMSE_max, sensitivity_max_total, specificity_max_total

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

def herd_trends(y_pred, x_test, y_test, scalar_y, plot=False):
    # get cow details
    cleaned_cows = ['8022015', '8022034', '8022073', '8022092', '8022445', '8026043', '8026045', '8026047', '8026066',
                    '8026106', '8026154', '8026216', '8026243', '8026280', '8026304', '8026319', '8026428', '8026499',
                    '8026522', '8026581', '8026585', '8026620', '8026621', '8026646', '8026668', '8026672', '8026699',
                    '8026873', '8026891', '8026915', '8026953', '8026962', '8026968', '8027009', '8027035', '8027091',
                    '8027097', '8027107', '8027181', '8027184', '8027186', '8027187', '8027207', '8027351', '8027462',
                    '8027464', '8027476', '8027551', '8027560', '8027596', '8027603', '8027633', '8027664', '8027686',
                    '8027688', '8027690', '8027716', '8027728', '8027752', '8027780', '8027781', '8027803', '8027808',
                    '8027813', '8027817', '8027945', '8027951', '8028001', '8028095', '8028101', '8028105', '8028132',
                    '8028177', '8028178', '8028186', '8028211', '8028217', '8028244', '8028255', '8028457', '8028565',
                    '8028603', '8028649', '8028654', '8028655', '8028776', '8028811', '8028867', '8029798', '8029859',
                    '8029865', '8029920', '8030585', '8032076', '8032104', '8032130', '8032154', '8032156', '8032169',
                    '8032183', '8032198', '8032212', '8032229', '8032237', '8032360', '8032383', '8032468', '8032473',
                    '8032494', '8032505', '8032506', '8032512', '8032524', '8032525', '8032526', '8032537', '8032541',
                    '8032550', '8033173', '8033175', '8033211', '8033214', '8033215', '8033222', '8033223', '8033238',
                    '8033246', '8033249', '8033251', '8033255', '8033275', '8033302', '8033306', '8033343', '8033348',
                    '8033450', '8038000', '8038882', '8038884', '8038896', '8038930', '8038943', '8039058', '8039064',
                    '8039075', '8039086', '8039093', '8039099', '8039101', '8039102', '8039116', '8039119', '8039131',
                    '8039139', '8039143', '8039148', '8039215', '8039768', '8039813', '8039920', '8040301', '8040458',
                    '8040459', '8040517', '8040638', '8040828', '8041081', '8042471', '8044725', '8044738', '8044842',
                    '8045166', '8045218', '8045228', '8045535', '8045629', '8045770', '8045813', '8045831', '8045858',
                    '8045911', '8045942', '8046335', '8046353', '8046559', '8046592', '8046782', '8047001', '8047033',
                    '8047122', '8047228', '8047389', '8047412', '8047516', '8047842', '8047983', '8048118']
    num_cows = len(cleaned_cows)
    cow_details_df = pd.read_csv('csvs_brisbane_18_10_18__2_1_19_tags/Cow_ID.csv')
    breed_dict = create_category_dict("Breed", "Tag#", cow_details_df)
    coat_dict = create_category_dict("Coat colour", "Tag#", cow_details_df)

    # only care about original test value for assessing error.
    x_test = x_test[0]
    samples_per_cow = int(x_test.shape[0] / num_cows)


    # errors per sample
    coat_errors = {"Red": [], "Black": [], "Tan": [], "White": []}
    breed_errors = {"39%": [], "0%": [], "50%": []}
    coat_daily_actual = {"Red": [], "Black": [], "Tan": [], "White": []}
    coat_daily_forecast = {"Red": [], "Black": [], "Tan": [], "White": []}
    breed_daily_actual = {"39%": [], "0%": [], "50%": []}
    breed_daily_forecast = {"39%": [], "0%": [], "50%": []}
    # for i in iter:
    for sample_n in range(0, samples_per_cow):
        # prediction dictionaries for this sample
        coat_prediction = {"Red": [], "Black": [], "Tan": [], "White": []}
        coat_actual = {"Red": [], "Black": [], "Tan": [], "White": []}
        breed_prediction = {"39%": [], "0%": [], "50%": []}
        breed_actual = {"39%": [], "0%": [], "50%": []}
        for cow_n in range(0, num_cows):
            # extract cow
            cow = cleaned_cows[cow_n]

            # extract prediction
            i = sample_n + cow_n * samples_per_cow
            y_pred_orig = scalar_y.inverse_transform(y_pred[i])
            test_y_i = y_test[i].reshape(-1, 1)
            y_actual_orig = scalar_y.inverse_transform(test_y_i)

            # add to relevant coat dict
            for coat in coat_dict.keys():
                if cow in set(str(x) for x in coat_dict[coat]):
                    # print(y_pred_orig)
                    # print(y_actual_orig)
                    coat_prediction[coat].append(y_pred_orig)
                    coat_actual[coat].append(y_actual_orig)

            # add to relevant breed dict
            for breed in breed_dict.keys():
                if cow in set(str(x) for x in breed_dict[breed]):
                    # print(y_pred_orig)
                    # print(y_actual_orig)
                    breed_prediction[breed].append(y_pred_orig)
                    breed_actual[breed].append(y_actual_orig)

        # take mean of prediction/actual and plot
        if plot:
            plt.figure()

        for coat in coat_prediction.keys():
            # calc means
            mean_pred = np.mean(np.array(coat_prediction[coat]), axis=0)
            mean_actual = np.mean(np.array(coat_actual[coat]), axis=0)
            # append error
            error = mean_squared_error(mean_actual, mean_pred, squared=False)
            coat_errors[coat].append(error)
            # update daily average dict
            if (sample_n+20)%24==0:
                coat_daily_actual[coat].append(mean_actual)
                coat_daily_forecast[coat].append(mean_pred)

            # plot individual
            if plot:
                plt.title('Coat Actual')
                # plt.plot(mean_pred, label=coat+' predicted')
                plt.plot(mean_actual, label=coat+' actual')
                plt.legend()

        if plot:
            plt.figure()
            for coat in coat_prediction.keys():
                mean_pred = np.mean(np.array(coat_prediction[coat]), axis=0)
                plt.title('Coat Predicted')
                plt.plot(mean_pred, label=coat+' predicted')
                # plt.plot(mean_actual, label=coat + ' actual')
                plt.legend()

        # take mean of prediction/actual and plot
        if plot:
            plt.figure()
        for breed in breed_prediction.keys():
            mean_pred = np.mean(np.array(breed_prediction[breed]), axis=0)
            mean_actual = np.mean(np.array(breed_actual[breed]), axis=0)
            # append error
            error = mean_squared_error(mean_actual, mean_pred, squared=False)
            breed_errors[breed].append(error)
            # update daily average dict
            if (sample_n + 20) % 24 == 0:
                breed_daily_actual[breed].append(mean_actual)
                breed_daily_forecast[breed].append(mean_pred)

            if plot:
                plt.title('Breed Actual')
                # plt.plot(mean_pred, label=breed+' predicted')
                plt.plot(mean_actual, label=breed+' actual')
                plt.legend()

        if plot:
            plt.figure()
            for breed in breed_prediction.keys():
                mean_pred = np.mean(np.array(breed_prediction[breed]), axis=0)
                plt.title('Breed Predicted')
                plt.plot(mean_pred, label=breed+' predicted')
                plt.legend()

        if plot:
            plt.show()

    # generate average prediction plots
    plt.figure()
    for breed in breed_daily_forecast.keys():
        mean_pred_daily = np.mean(np.array(breed_daily_forecast[breed]), axis=0)
        plt.title('Breed Predicted Average')
        plt.plot(mean_pred_daily, label=breed + ' predicted')
        plt.xlabel("Hour of the Day")
        plt.ylabel("Average Filtered Panting")
        plt.legend()

    plt.figure()
    for breed in breed_daily_actual.keys():
        mean_actual_daily = np.mean(np.array(breed_daily_actual[breed]), axis=0)
        plt.title('Breed Actual Average')
        plt.plot(mean_actual_daily, label=breed + ' actual')
        plt.xlabel("Hour of the Day")
        plt.ylabel("Average Filtered Panting")
        plt.legend()

    plt.figure()
    for coat in coat_daily_forecast.keys():
        mean_pred_daily = np.mean(np.array(coat_daily_forecast[coat]), axis=0)
        plt.title('Coat Predicted Average')
        plt.plot(mean_pred_daily, label=coat + ' predicted')
        plt.xlabel("Hour of the Day")
        plt.ylabel("Average Filtered Panting")
        plt.legend()

    plt.figure()
    for coat in coat_daily_actual.keys():
        mean_actual_daily = np.mean(np.array(coat_daily_actual[coat]), axis=0)
        plt.title('Coat Actual Average')
        plt.plot(mean_actual_daily, label=coat + ' actual')
        plt.xlabel("Hour of the Day")
        plt.ylabel("Average Filtered Panting")
        plt.legend()
    plt.show()

    print("\n\nErrors")
    error_list = []
    for coat in coat_errors.keys():
        mean_error = np.mean(np.array(coat_errors[coat]))
        error_list.append(mean_error)
        print(coat + ": " + str(mean_error))

    for breed in breed_errors.keys():
        mean_error = np.mean(np.array(breed_errors[breed]))
        error_list.append(mean_error)
        print(breed + ": " + str(mean_error))

    mean_herd_error = np.mean(error_list)
    print("Herd: " + str(mean_herd_error))
    print('\n\n')

    return mean_herd_error

def edit_num_lags(train_x, test_x, new_lag):
    train_x[0] = reduce_lags(new_lag, train_x[0])
    test_x[0] = reduce_lags(new_lag, test_x[0])
    return train_x, test_x

def reduce_lags(new_lag, l):
    l = list(l)
    for i in range(len(l)):
        l[i] = l[i][-new_lag:]
    return np.array(l)

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

def baseline_individual_errors(model_location, data_location, lag, name='', forecast_error = True, post_process = False, skip_4_fold= False):
    batch_size = 128
    epochs = 75

    forecast_error_summary = []
    post_process_error = []

    model_name = model_location + '/' + name + '-batch_size' + str(128) + '-' + str(75) + '.hdf5'

    # extract model data
    model, x_train, y_train, x_test, y_test, scalar_y = import_model(data_location, model_name)

    # predict
    # x_test[0] only when no forecasts considered
    print("Making Predictions")
    y_pred = []
    for sample in x_train[0]:
        prev_days_panting = []
        for j in range(0,4):
            pant = []
            for i in range(-24-j*24,0-j*24):
                pant.append([sample[i][0]])
            prev_days_panting.append(pant)
        y_pred.append(np.mean(prev_days_panting, axis=0))

    print("Calculating Error")
    if forecast_error:
        mean_hourly_RMSE, median_hourly_RMSE, RMSE_daily_freq, thresh_count_RMSE, sensitivity_freq_total, specificity_freq_total, RMSE_max, sensitivity_max_total, specificity_max_total = test_error(y_pred, x_test, y_test, scalar_y, plot=False, skip_4_fold=skip_4_fold)
        forecast_error_summary.append([batch_size, epochs, mean_hourly_RMSE, median_hourly_RMSE, RMSE_daily_freq, thresh_count_RMSE, sensitivity_freq_total, specificity_freq_total, RMSE_max, sensitivity_max_total, specificity_max_total])

    if post_process:
        sens_old, spec_old, sens_new, spec_new, pos_perc_max, pos_perc_count = max_count_threshold_model(x_train,
                                                                                                         y_train,
                                                                                                         x_test,
                                                                                                         y_test,
                                                                                                         model,
                                                                                                         scalar_y,
                                                                                                         learning_rate=0.001,
                                                                                                         batch_size=16,
                                                                                                         epochs=10,
                                                                                                         thresh_pant=12,
                                                                                                         thresh_count=4,
                                                                                                         test=True)

        post_process_error.append([batch_size, epochs, sens_old, spec_old, sens_new, spec_new, pos_perc_max, pos_perc_count])

    if forecast_error:
        error_summary_df = pd.DataFrame(forecast_error_summary, columns=['batch size', 'epochs', 'mean hourly RMSE', 'median hourly RMSE', 'daily freq RMSE', 'thresh_count_RMSE', 'thresh sensitivity', 'thresh specificity', 'max RMSE', 'max sensitivity', 'max specificity'])
        print(error_summary_df)
        error_summary_df.to_pickle(model_location + '/individual_error_summary.pkl')

    if post_process:
        post_process_summary_df = pd.DataFrame(post_process_error, columns=['batch size', 'epochs', 'old sens', 'old spec', 'new sens', ' new spec', '% pos max', '% pos count'])
        print(post_process_summary_df)
        post_process_summary_df.to_pickle(model_location + '/post_process_error_summary.pkl')

def compare_model_individual_errors(model_location, data_location, lag, name='', forecast_error = True, post_process = False, skip_4_fold=False):
    forecast_error_summary = []
    post_process_error = []
    for batch_size in [128]:
        # for epochs in [50, 100, 150]:
        for epochs in [25, 50, 75, 100]:
        # for epochs in [75]:
            # try:
            print('\nBatch Size: ' + str(batch_size))
            print('Epochs: ' + str(epochs))

            model_name = model_location + '/' + name + '-batch_size' + str(batch_size) + '-' + str(epochs) + '.hdf5'

            try:
                # extract model data
                model, x_train, y_train, x_test, y_test, scalar_y = import_model(data_location, model_name)
            except Exception as e:
                print(e)
                continue

            # reduce number of lags
            if lag != 200:
                x_train, x_test = edit_num_lags(x_train, x_test, lag)

            # change to uni or bivariate data
            if name[0:10] == 'univariate':
                x_train, x_test = convert_to_univariate(x_train, x_test)
            if name[0:9] == 'bivariate':
                x_train, x_test = convert_to_bivariate(x_train, x_test)

            # predict
            print("Making Predictions")
            if name[0:9] == 'iterative':
                x_train, x_test = convert_to_univariate(x_train, x_test)
                y_pred = []
                y_test_new = []
                # need to also update x_test
                # need to account for change in animals
                samples_per_cow = int(len(x_test[0])/197)
                for cow in range(197):
                    print(cow)
                    for i in range(samples_per_cow-24):
                        index = cow*samples_per_cow + i
                        panting_series = np.array([x_test[0][index]])
                        y_test_new.append(y_test[index])
                        # print(panting_series)
                        for j in range(24):
                            x_test_sample = [panting_series, np.array([x_test[1][index+j]])]
                            y_pred_sample = model.predict(x_test_sample)
                            # print(panting_series)
                            # print(y_pred_sample[0][0])
                            panting_series = np.delete(panting_series,0)
                            panting_series = np.append(panting_series,y_pred_sample)
                            panting_series = panting_series.reshape((1,120,1))
                            # print(panting_series)
                        # print(panting_series)
                        pred = panting_series[0,-24:]
                        # print(pred)
                        y_pred.append(pred)
                        # plt.figure()
                        # plt.plot(pred)
                        # plt.plot(y_test[i])
                        # plt.show()
                y_test = np.array(y_test_new)
            else:
                # x_test[0] only when no forecasts considered
                y_pred = model.predict(x_test)

            print("Calculating Error")
            if forecast_error:
                mean_hourly_RMSE, median_hourly_RMSE, mean_r_squared, RMSE_daily_freq, thresh_count_RMSE, sensitivity_freq_total, specificity_freq_total, RMSE_max, sensitivity_max_total, specificity_max_total = test_error(y_pred, x_test, y_test, scalar_y, plot=False, skip_4_fold=skip_4_fold)
                forecast_error_summary.append([batch_size, epochs, mean_hourly_RMSE, median_hourly_RMSE,mean_r_squared, RMSE_daily_freq, thresh_count_RMSE, sensitivity_freq_total, specificity_freq_total, RMSE_max, sensitivity_max_total, specificity_max_total])

            if post_process:
                sens_old, spec_old, sens_new, spec_new, pos_perc_max, pos_perc_count = max_count_threshold_model(x_train,
                                                                                                                 y_train,
                                                                                                                 x_test,
                                                                                                                 y_test,
                                                                                                                 model,
                                                                                                                 scalar_y,
                                                                                                                 learning_rate=0.001,
                                                                                                                 batch_size=16,
                                                                                                                 epochs=10,
                                                                                                                 thresh_pant=12,
                                                                                                                 thresh_count=4,
                                                                                                                 test=True)

                post_process_error.append([batch_size, epochs, sens_old, spec_old, sens_new, spec_new, pos_perc_max, pos_perc_count])

            # except OSError:
            # except Exception as e:
            #     print(e)
            #     continue

    if forecast_error:
        error_summary_df = pd.DataFrame(forecast_error_summary, columns=['batch size', 'epochs', 'mean hourly RMSE', 'median hourly RMSE', 'r squared','daily freq RMSE', 'thresh_count_RMSE', 'thresh sensitivity', 'thresh specificity', 'max RMSE', 'max sensitivity', 'max specificity'])
        print(error_summary_df)
        path = '/individual_error_summary.pkl'
        error_summary_df.to_pickle(model_location + path)

    if post_process:
        post_process_summary_df = pd.DataFrame(post_process_error, columns=['batch size', 'epochs', 'old sens', 'old spec', 'new sens', ' new spec', '% pos max', '% pos count'])
        print(post_process_summary_df)
        post_process_summary_df.to_pickle(model_location + '/post_process_error_summary.pkl')

def plot_timeseries_error(model_path, data_location, lag, model_name):
    # extract model data
    model, x_train, y_train, x_test, y_test, scalar_y = import_model(data_location, model_path)

    # reduce number of lags
    if lag != 200:
        x_train, x_test = edit_num_lags(x_train, x_test, lag)

    # change to uni or bivariate data
    if model_name[0:10] == 'univariate':
        x_train, x_test = convert_to_univariate(x_train, x_test)
    if model_name[0:9] == 'bivariate':
        x_train, x_test = convert_to_bivariate(x_train, x_test)

    # predict
    print("Making Predictions")
    y_pred = model.predict(x_test)

    # timeseries error
    error_timeseries, error_timeseries_ignore, weather_timeseries = timeseries_error(y_pred, x_test, y_test, x_train, scalar_y, plot=False)
    # cow error
    cow_error_list = calc_cow_errors(y_pred, x_test, y_test, x_train, scalar_y, plot=False)

    plt.figure()
    plt.plot(error_timeseries, 'bo')
    plt.title(model_name + ': Average Error Over Time')

    plt.figure()
    plt.plot(error_timeseries_ignore, 'bo', label = 'RMSE')
    plt.plot(weather_timeseries, label= 'HLI')
    plt.title(model_name + ': Average Error Over Time')

    plt.figure()
    plt.plot(cow_error_list, 'bo')
    n = [x for x in range(len(cow_error_list))]
    for i, txt in enumerate(n):
        plt.annotate(txt, (i, cow_error_list[i]))
    plt.title("Average Error per Cow")
    plt.show()

def compare_model_herd_errors(model_location, data_location, name = ''):
    error_summary = []

    for batch_size in [64,128, 256, 512]:
        for epochs in [50, 100, 150, 200, 250, 300]:

            try:
                print('\nBatch Size: ' + str(batch_size))
                print('Epochs: ' + str(epochs))
                model_name = model_location + '/' + name + '-batch_size' + str(batch_size) + '-' + str(epochs) + '.hdf5'

                # extract model data
                model, x_train, y_train, x_test, y_test, scalar_y = import_model(data_location, model_name)

                # predict
                print("Making Predictions")
                y_pred = model.predict(x_test)

                print("Calculating Error")
                mean_hourly_RMSE, RMSE_daily_freq, thresh_count_RMSE, sensitivity_freq_total, specificity_freq_total, RMSE_max, sensitivity_max_total, specificity_max_total,  = test_error_herd(y_pred, x_test, y_test, scalar_y, plot=False)

                error_summary.append([batch_size, epochs, mean_hourly_RMSE, RMSE_daily_freq, thresh_count_RMSE, sensitivity_freq_total, specificity_freq_total, RMSE_max, sensitivity_max_total, specificity_max_total])
            except OSError:
                continue

    error_summary_df = pd.DataFrame(error_summary, columns=['batch size', 'epochs', 'mean hourly RMSE', 'daily freq RMSE', 'thresh_count_RMSE', 'thresh sensitivity', 'thresh specificity', 'max RMSE', 'max sensitivity', 'max specificity'])
    print(error_summary_df)
    error_summary_df.to_pickle(model_location + '/herd_error_summary.pkl')

def plot_model_error(error_df, epochs, errors, herd_ind):
    for error_type in errors:
        # plot for different epochs
        plt.figure()
        for epoch in epochs:
            epoch_error = error_df[error_df['epochs']==epoch]
            x = epoch_error['batch size'].values
            y = epoch_error[error_type].values
            plt.title(herd_ind + " " + error_type)
            plt.xlabel('batch size')
            plt.ylabel('error')
            plt.plot(x,y,label = 'epoch: ' + str(epoch))
        plt.legend()
    plt.show()

def test_individual_model(model_location, data_location):
    # extract model data
    model, x_train, y_train, x_test, y_test, scalar_y = import_model(data_location, model_location)

    # predict
    print("Making Predictions")
    y_pred = model.predict(x_test)

    print("Inidividual Error")
    test_error(y_pred, x_test, y_test, scalar_y, plot=False)

    print("Herd Error")
    test_error_herd(y_pred, x_test, y_test, scalar_y, plot=False)

def plot_subherd_trends(model_location, data_location):
    # extract model data
    model, x_train, y_train, x_test, y_test, scalar_y = import_model(data_location, model_location)

    # predict
    print("Making Predictions")
    y_pred = model.predict(x_test)

    # run herd trends
    herd_trends(y_pred, x_test, y_test, scalar_y, plot=False)

def plot_model_error(error_df, epochs, errors, herd_ind):
    for error_type in errors:
        # plot for different epochs
        plt.figure()
        for epoch in epochs:
            epoch_error = error_df[error_df['epochs']==epoch]
            x = epoch_error['batch size'].values
            y = epoch_error[error_type].values
            plt.title(herd_ind + " " + error_type)
            plt.xlabel('batch size')
            plt.ylabel('error')
            plt.plot(x,y,label = 'epoch: ' + str(epoch))
        plt.legend()
    plt.show()

def plot_n_fold_error(error_df_n_folds, epochs, batch_sizes, errors, herd_ind):

    for error_type in errors:
        # plot for different epochs
        plt.figure()
        for epoch in epochs:
            dummy = error_df_n_folds['1_fold'][error_df_n_folds['1_fold']['epochs']==epoch]
            for batch_size in batch_sizes:
                print(epoch)
                print(batch_size)
                error_list = []
                n_fold_list = []
                for n_fold, error_df in error_df_n_folds.items():
                    epoch_error = error_df[(error_df['epochs']==epoch) & (error_df['batch size']==batch_size)]
                    print(epoch_error)
                    error_list.append(epoch_error[error_type])
                    n_fold_list.append(n_fold + " (" + str(n_fold_perc_pos[n_fold]) + ")")
                plt.title("n_fold " + error_type)
                plt.xlabel('n_fold')
                plt.ylabel('error')
                plt.plot(n_fold_list, error_list,label = 'epoch: ' + str(epoch) + ', batch_size: ' + str(batch_size))
        plt.legend()
    plt.show()

def compare_models_n_fold(model_paths, batch_sizes, epochs, errors, model_names, error_dict_name, test_name = 'Default Test'):
    model_error_dfs = {}
    for i in range(len(model_paths)):
        model_path = model_paths[i]
        model_name = model_names[i]
        model_error_dfs[model_name] = {}
        for fold in ['1_fold', '2_fold', '3_fold', '4_fold', '5_fold']:
            try:
                ind_error_df = pd.read_pickle(model_path + fold + '/' + error_dict_name)
                # ensure df is sorted so plots look nice
                ind_error_df = ind_error_df.sort_values(by=['batch size', 'epochs'])
                # n_fold_error_df
                model_error_dfs[model_name][fold] = ind_error_df
            except Exception as e:
                print(e)
                continue

    for error_type in errors:
        # plot for different epochs
        plt.figure()

        # iterate through two models
        i = 0
        for model_name, n_fold_error_df in model_error_dfs.items():
            # extract model 1 epoch
            epoch = epochs[i]
            batch_size = batch_sizes[i]
            print(epoch)
            print(batch_size)

            # build errors to plot
            error_list = []
            n_fold_list = []
            for n_fold, error_df in n_fold_error_df.items():
                epoch_error = error_df[(error_df['epochs']==epoch) & (error_df['batch size']==batch_size)]
                print(epoch_error[error_type])
                error_list.append(epoch_error[error_type])
                # n_fold_list.append(n_fold + " (" + str(n_fold_perc_pos[n_fold]) + ")")
                n_fold_list.append(n_fold[0])
            plt.title(error_type)
            plt.title(error_type.title().replace('Rmse', 'RMSE') + ' for ' + test_name)
            plt.xlabel('Fold')
            plt.ylabel('RMSE')
            plt.plot(n_fold_list, error_list,label = model_name)
            plt.legend()

            # increase index
            i+=1
    plt.show()


########################## PAPER MODELLING ###############################################
# n_fold percentage positive:
n_fold_perc_pos = {'1_fold': 0.244, '2_fold': 0.384, '3_fold': 0.487, '4_fold': 0.356, '4_fold_skip': 0.206, '5_fold': 0.323}

# creates a dictionary of all individual errors of multiple models and saves it to file
skip_4_fold = False
for fold in ['1_fold', '2_fold', '3_fold', '4_fold', '5_fold']:
# for fold in ['4_fold_skip']:
    print(fold)
    # declare name of models
    name = 'iterative_'
    # augment the skip
    if fold == '4_fold_skip':
        skip_4_fold = True
    # baseline_individual_errors('LSTM Models/Baseline/' + fold , 'Deep Learning Data/Multivariate Lag 200/' + fold[0:6], 120, 'no_forecast_' + fold[0:6], forecast_error=True, post_process=False, skip_4_fold = skip_4_fold)
    compare_model_individual_errors('LSTM Models/Iterative Pred/Univariate/' + fold, 'Deep Learning Data/Multivariate Lag 200/' + fold[0:6], 120, name + fold[0:6], forecast_error=True, post_process=False, skip_4_fold=skip_4_fold)
    # compare_model_individual_errors('LSTM Models/Lag Test/Univariate/150/' + fold, 'Deep Learning Data/Multivariate Lag 200/' + fold[0:6], 150, name + fold[0:6], forecast_error=True, post_process=False, skip_4_fold=skip_4_fold)

n_fold_error_df = {}
for fold in ['1_fold', '2_fold', '3_fold', '4_fold', '5_fold']:
    ind_error_df = pd.read_pickle('LSTM Models/Lag Test/Univariate/150/' + fold + '/individual_error_summary.pkl')
    # ensure df is sorted so plots look nice
    ind_error_df = ind_error_df.sort_values(by=['batch size', 'epochs'])
    # n_fold_error_df
    n_fold_error_df[fold] = ind_error_df

# plot errors
# plot_n_fold_error(n_fold_error_df, [75], [64, 128, 256], ['mean hourly RMSE', 'median hourly RMSE'], fold)
# plot_n_fold_error(n_fold_error_df, [25, 50, 75, 100], [128], ['mean hourly RMSE', 'median hourly RMSE', 'r squared'], fold)

# plot entire workflow errors:
# plot_n_fold_error(n_fold_error_df, [100], [64, 128, 256], ['old sens', 'new sens', 'old spec', ' new spec'], fold)

# compare models from different tests
# Lag tests
# analysis of R squared. Potentially include in all tests to show overall performance.
# compare_models_n_fold(['LSTM Models/Pooling Test/pool all forecast/', 'LSTM Models/Lag Test/Multivariate/90/', 'LSTM Models/Lag Test/Univariate/150/'], [128, 128, 128], [75, 25, 50], ['mean hourly RMSE', 'median hourly RMSE', 'r squared'], ['pool all', '90_lag', '150_lag_uni'], 'individual_error_summary.pkl')


# plot the timeseries and cattle error across an n_fold test
# for fold in ['4_fold']:
#     plot_timeseries_error('LSTM Models/Pooling Test/pool all forecast/' + fold + '/no_forecast_' + fold + '-batch_size128-75.hdf5', 'Deep Learning Data/Multivariate Lag 200/' + fold, 120, 'Optimal Multivariate')

# paper compare model tests
# # crude grid search
# compare_models_n_fold(['LSTM Models/Pooling Test/pool all forecast/', 'LSTM Models/input n_fold 2/multivariate/', 'LSTM Models/input n_fold 2/multivariate/'], [128, 256, 64], [75, 75, 50], ['mean hourly RMSE', 'median hourly RMSE'], ['128', '256', '64'], 'individual_error_summary.pkl')
# # pooling method
# compare_models_n_fold(['LSTM Models/Pooling Test/no forecast/', 'LSTM Models/Pooling Test/post pooling/', 'LSTM Models/Pooling Test/pool all forecast/'], [128, 128, 128], [25, 75, 75], ['mean hourly RMSE', 'median hourly RMSE'], ['No Forecast', 'Post Pooling', 'Mid Pooling'], 'individual_error_summary.pkl')
# # input data type
# compare_models_n_fold(['LSTM Models/input n_fold 2/bivariate/', 'LSTM Models/Pooling Test/pool all forecast/', 'LSTM Models/input n_fold 2/univariate/'], [128, 128, 128], [75, 75, 50], ['mean hourly RMSE', 'median hourly RMSE'], ['Bivariate', 'Multivariate', 'Univariate'], 'individual_error_summary.pkl')
# # multivariate lag
# compare_models_n_fold(['LSTM Models/Pooling Test/pool all forecast/', 'LSTM Models/Lag Test/multivariate/90/', 'LSTM Models/Lag Test/multivariate/60/'], [128, 128, 128], [75, 25, 100], ['mean hourly RMSE', 'median hourly RMSE'], ['Multivariate 120 Lag', 'Mutlivariate 90 Lag', 'Multivariate 60 Lag'], 'individual_error_summary.pkl', 'Different Lags')
# # univariate lag
# compare_models_n_fold(['LSTM Models/Lag Test/Univariate/150/', 'LSTM Models/input n_fold 2/univariate/'], [128, 128], [50, 50], ['mean hourly RMSE', 'median hourly RMSE'], ['Univariate Lag 150', 'Univariate Lag 120'], 'individual_error_summary.pkl', 'Different Lags')

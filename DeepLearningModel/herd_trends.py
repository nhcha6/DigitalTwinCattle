import pandas as pd
import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from matplotlib import pyplot as plt
from keras import models
import pickle
import random
from math import floor, ceil
from keras.initializers import Orthogonal

def import_model(test_file, model_file):
    x_train, y_train, x_test, y_test, scalar_y = read_pickle(test_file)

    print("loading model")
    model = models.load_model(model_file)

    return model, x_train, y_train, x_test, y_test, scalar_y

def edit_num_lags(train_x, test_x, new_lag):
    train_x = reduce_lags(new_lag, train_x)
    test_x = reduce_lags(new_lag, test_x)
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

def create_category_dict(category_header, cow_ID_header, details_df):
    categories = set(details_df[category_header])
    category_dict = {key: [] for key in categories}
    for category in categories:
        category_dict[category] = details_df[cow_ID_header][details_df[category_header] == category].tolist()
    #print(category_dict)
    return category_dict

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

def edit_train_test(x_train, y_train, x_test, y_test, horizon, lag, num_cows=197):
    # remove first animal as it has a 3 day period of invalid data
    x_train = x_train[899:]
    y_train = y_train[899:]
    x_test = x_test[604:]
    y_test = y_test[604:]

    # smaller prediction horizon
    if horizon != 24:
        y_train = y_train[:,0:horizon]
        y_test = y_test[:,0:horizon]

    if lag!=200:
        x_train, x_test = edit_num_lags(x_train, x_test, lag)

    x_train, y_train, x_test, y_test = add_forecast_input(x_train, y_train, x_test, y_test, num_cows)

    return x_train, y_train, x_test, y_test

def test_error(y_pred, test_x, test_y, norm_y, num_cows=197, invert_test = [],y_prev=0, model_type='forecast', plot=False):

    # only care about original test value for assessing error.
    if model_type=='forecast':
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
            # norm_error = error/max(y_actual_orig)
            norm_error = error

            # calculate frequencies
            freq_actual = sum(y_actual_orig)
            freq_forecast = sum(y_pred_orig)
            max_actual = max(y_actual_orig)
            max_forecast = max(y_pred_orig)

            # error in frequency prediction
            daily_error = abs(freq_actual - freq_forecast)
            freq_actual_dict[cow_n] = freq_actual
            freq_forecast_dict[cow_n] = freq_forecast
            max_error = abs(max_actual - max_forecast)

            # if max(y_actual_orig) > 1:
            if True:
                hourly_errors.append(norm_error)
                daily_errors.append(daily_error)
                max_errors.append(max_error)

            # calculate false positive and false negative above and below daily threshold
            if freq_actual > 158:
                total_pos_freq += 1
                if freq_forecast < 158:
                    # plot = True
                    false_neg_freq += 1
            else:
                total_neg_freq += 1
                if freq_forecast > 158:
                    # plot = True
                    false_pos_freq += 1

            # calculate false positive and false negative above and below daily threshold
            if max_actual > 12:
                total_pos_max += 1
                if max_forecast < 12:
                    # plot = True
                    false_neg_max += 1
            else:
                total_neg_max += 1
                if max_forecast > 12:
                    # plot = True
                    false_pos_max += 1

            if plot:
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

    # calculate summary errors
    mean_hourly_RMSE = np.mean(hourly_errors)
    RMSE_daily_freq = np.mean(daily_errors)
    RMSE_max = np.mean(max_errors)
    false_pos_freq_total = false_pos_freq/total_neg_freq
    false_neg_freq_total = false_neg_freq/total_pos_freq
    false_pos_max_total = false_pos_max/total_neg_max
    false_neg_max_total = false_neg_max/total_pos_max

    print("\n")
    print("\n")
    print("\nMEAN HOURLY RMSE")
    print(mean_hourly_RMSE)
    print("\nRMSE DAILY FREQ")
    print(RMSE_daily_freq)
    print("\nRMSE MAX ERROR")
    print(RMSE_max)
    # print("\nMEAN TOP 20 PREDICTED")
    # print(np.mean(predictions))
    print("\nFALSE POS DAILY FREQ")
    print(false_pos_freq_total)
    print("\nFALSE NEG DAILY FREQ")
    print(false_neg_freq_total)
    print("\nFALSE POS MAX")
    print(false_pos_max_total)
    print("\nFALSE NEG MAX")
    print(false_neg_max_total)
    print("\n")
    print("\n")

    return mean_hourly_RMSE, RMSE_daily_freq, false_pos_freq_total, false_neg_freq_total, RMSE_max, false_pos_max_total, false_neg_max_total

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
    cow_details_df = pd.read_csv('Cow_ID.csv')
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
        plt.legend()

    plt.figure()
    for breed in breed_daily_actual.keys():
        mean_actual_daily = np.mean(np.array(breed_daily_actual[breed]), axis=0)
        plt.title('Breed Actual Average')
        plt.plot(mean_actual_daily, label=breed + ' predicted')
        plt.legend()

    plt.figure()
    for coat in coat_daily_forecast.keys():
        mean_pred_daily = np.mean(np.array(coat_daily_forecast[coat]), axis=0)
        plt.title('Coat Predicted Average')
        plt.plot(mean_pred_daily, label=coat + ' predicted')
        plt.legend()

    plt.figure()
    for coat in coat_daily_actual.keys():
        mean_actual_daily = np.mean(np.array(coat_daily_actual[coat]), axis=0)
        plt.title('Coat Actual Average')
        plt.plot(mean_actual_daily, label=coat + ' predicted')
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

def compare_model_errors():
    error_summary = []
    for batch_size in [128]:
        # for epochs in [50]:
        for epochs in [50, 100, 150, 200, 250, 300]:
            print('\nBatch Size: ' + str(batch_size))
            print('Epochs: ' + str(epochs))
            model_name = 'model checkpoints/batch_size' + str(batch_size) + '-' + str(epochs) + '.hdf5'

            # extract model data
            model, x_train, y_train, x_test, y_test, scalar_y = import_model('normalised lag 120', model_name)

            # predict
            print("Making Predictions")
            y_pred = model.predict(x_test)

            # edit data to suit model
            # x_train, y_train, x_test, y_test = edit_train_test(x_train, y_train, x_test, y_test, 24, 120)
            # print('writing data')
            # write_pickle(x_train, y_train, x_test, y_test, scalar_y)

            print("Entire Error")
            mean_hourly_RMSE, RMSE_daily_freq, false_pos_freq_total, false_neg_freq_total, RMSE_max, false_pos_max_total, false_neg_max_total = test_error(y_pred, x_test, y_test, scalar_y, plot=False)

            # print("Error Ignoring First 6 Hours")
            # test_error(y_pred, x_test, y_test, scalar_y, y_prev=6)

            print("Herd Error")
            mean_herd_error = herd_trends(y_pred, x_test, y_test, scalar_y, False)

            error_summary.append([batch_size, epochs, mean_hourly_RMSE, RMSE_daily_freq, false_pos_freq_total, false_neg_freq_total, RMSE_max, false_pos_max_total, false_neg_max_total, mean_herd_error])

    error_summary_df = pd.DataFrame(error_summary, columns=['batch size', 'epochs', 'mean hourly RMSE', 'daily freq RMSE', 'daily freq fp', 'daily freq fn', 'max RMSE', 'max fp', 'max fn', 'mean herd RMSE'])
    print(error_summary_df)
    error_summary_df.to_pickle('model checkpoints/error_summary_2.pkl')

def plot_model_error(error_df):
    for error_type in ['mean hourly RMSE', 'daily freq RMSE', 'daily freq fp', 'daily freq fn', 'max RMSE', 'max fp','max fn', 'mean herd RMSE']:
        plt.figure()
        # plot for different epochs
        for batch_size in [128,256,512]:
            batch_error = error_df[error_df['batch size']==batch_size]
            x = batch_error['epochs'].values
            y = batch_error[error_type].values
            plt.title(error_type)
            plt.plot(x,y,label = 'batch size: ' + str(batch_size))
        plt.legend()

        # plot for different epochs
        plt.figure()
        for epoch in [50, 100, 150, 200, 250, 300]:
            epoch_error = error_df[error_df['epochs']==epoch]
            x = epoch_error['batch size'].values
            y = epoch_error[error_type].values
            plt.title(error_type)
            plt.plot(x,y,label = 'epoch: ' + str(epoch))
        plt.legend()


    plt.show()

def summarise_herd_trends(model_name):
    # extract model data
    model, x_train, y_train, x_test, y_test, scalar_y = import_model('normalised lag 120', model_name)

    # predict
    print("Making Predictions")
    y_pred = model.predict(x_test)

    print("Herd Error")
    mean_herd_error = herd_trends(y_pred, x_test, y_test, scalar_y, False)


# summarise_herd_trends('model checkpoints/batch_size256-100.hdf5')

# compare_model_errors()

error_df = pd.read_pickle('model checkpoints/error_summary.pkl')
error_df = error_df[(error_df['batch size']!=128) | (error_df['epochs']!=200)]
error_df = error_df.sort_values(by=['batch size', 'epochs'])
with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(error_df)
# error_df = pd.concat([error_df, pd.read_pickle('model checkpoints/error_summary_2.pkl')])
# error_df.to_pickle('model checkpoints/error_summary.pkl')
plot_model_error(error_df)
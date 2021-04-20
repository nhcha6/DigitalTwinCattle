import pandas as pd
import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from matplotlib import pyplot as plt
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

def create_category_dict(category_header, cow_ID_header, details_df):
    categories = set(details_df[category_header])
    category_dict = {key: [] for key in categories}
    for category in categories:
        category_dict[category] = details_df[cow_ID_header][details_df[category_header] == category].tolist()
    #print(category_dict)
    return category_dict

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

    print(total_pos_freq)
    print(total_pos_max)

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

def compare_model_individual_errors(model_location, data_location):
    error_summary = []
    for batch_size in [64,128, 256, 512]:
        for epochs in [50, 100, 150, 200, 250, 300]:
            try:
                print('\nBatch Size: ' + str(batch_size))
                print('Epochs: ' + str(epochs))
                model_name = model_location + '/multivariate-batch_size' + str(batch_size) + '-' + str(epochs) + '.hdf5'

                # extract model data
                model, x_train, y_train, x_test, y_test, scalar_y = import_model(data_location, model_name)

                # predict
                print("Making Predictions")
                y_pred = model.predict(x_test)

                print("Calculating Error")
                mean_hourly_RMSE, RMSE_daily_freq, thresh_count_RMSE, sensitivity_freq_total, specificity_freq_total, RMSE_max, sensitivity_max_total, specificity_max_total = test_error(y_pred, x_test, y_test, scalar_y, plot=False)

                error_summary.append([batch_size, epochs, mean_hourly_RMSE, RMSE_daily_freq, thresh_count_RMSE, sensitivity_freq_total, specificity_freq_total, RMSE_max, sensitivity_max_total, specificity_max_total])
            except OSError:
                continue

    error_summary_df = pd.DataFrame(error_summary, columns=['batch size', 'epochs', 'mean hourly RMSE', 'daily freq RMSE', 'thresh_count_RMSE', 'thresh sensitivity', 'thresh specificity', 'max RMSE', 'max sensitivity', 'max specificity'])
    print(error_summary_df)
    error_summary_df.to_pickle(model_location + '/individual_error_summary.pkl')

def compare_model_herd_errors(model_location, data_location):
    error_summary = []

    for batch_size in [64,128, 256, 512]:
        for epochs in [50, 100, 150, 200, 250, 300]:

            try:
                print('\nBatch Size: ' + str(batch_size))
                print('Epochs: ' + str(epochs))
                model_name = model_location + '/multivariate-batch_size' + str(batch_size) + '-' + str(epochs) + '.hdf5'

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


################## CREATE ERROR DF FOR MULTIVARIATE TEST ######################

# creates a dictionary of all individual errors of multiple models and saves it to file
# compare_model_individual_errors('LSTM Models/Multivariate Optimisation 2', 'Deep Learning Data/Multivariate Lag 120')

# creates a dictionary of all herd errors of multiple models and saves it to file
# compare_model_herd_errors('LSTM Models/Multivariate Optimisation 2', 'Deep Learning Data/Multivariate Lag 120')

################## CREATE ERROR DF FOR UNIVARIATE TEST ######################

# creates a dictionary of all individual errors of multiple models and saves it to file
# compare_model_individual_errors('LSTM Models/Univariate Optimisation', 'Deep Learning Data/Univariate Lag 120')

# creates a dictionary of all herd errors of multiple models and saves it to file
# compare_model_herd_errors('LSTM Models/Univariate Optimisation', 'Deep Learning Data/Univariate Lag 120')

################## PLOT ERROR FOR MULTIVARIATE TEST ######################

# read in error summary
# ind_error_df = pd.read_pickle('LSTM Models/Multivariate Optimisation 2/individual_error_summary.pkl')
herd_error_df = pd.read_pickle('LSTM Models/Multivariate Optimisation 2/herd_error_summary.pkl')
# # remove unstable error metrics (found by inspection)
herd_error_df = herd_error_df[(herd_error_df['batch size']!=64) | (herd_error_df['epochs']!=150)]
herd_error_df = herd_error_df[(herd_error_df['batch size']!=64) | (herd_error_df['epochs']!=250)]
herd_error_df = herd_error_df[(herd_error_df['batch size']!=64) | (herd_error_df['epochs']!=300)]
herd_error_df = herd_error_df[(herd_error_df['batch size']!=128) | (herd_error_df['epochs']!=200)]
herd_error_df = herd_error_df[(herd_error_df['batch size']!=200)]

# herd_error_df = herd_error_df[(herd_error_df['batch size']!=128) | (herd_error_df['epochs']!=200)]
# ind_error_df = ind_error_df[(ind_error_df['batch size']!=64) | (ind_error_df['epochs']!=150)]
# herd_error_df = herd_error_df[(herd_error_df['batch size']!=64) | (herd_error_df['epochs']!=150)]
# # ensure df is sorted so plots look nice
# ind_error_df = ind_error_df.sort_values(by=['batch size', 'epochs'])
# herd_error_df = herd_error_df.sort_values(by=['batch size', 'epochs'])
# plot errors
# plot_model_error(ind_error_df, [50, 100, 150, 200, 250, 300], ['mean hourly RMSE'], "Multivariate - Inidividual")
# plot_model_error(herd_error_df, [50, 100, 150, 200, 250, 300], ['mean hourly RMSE'], "Multivariate - Herd")
# plot_model_error(ind_error_df, [50, 100, 150], ['mean hourly RMSE', 'daily freq RMSE', 'thresh_count_RMSE', 'thresh sensitivity', 'thresh specificity', 'max RMSE', 'max sensitivity', 'max specificity'], "Multivariate - Inidividual")
plot_model_error(herd_error_df, [50, 100, 150], ['mean hourly RMSE', 'daily freq RMSE', 'thresh_count_RMSE', 'thresh sensitivity', 'thresh specificity', 'max RMSE', 'max sensitivity', 'max specificity'], "Multivariate - Herd")

####################### PLOT ERROR FOR UNIIVARIATE TEST ######################

# read in error summary
# ind_error_df = pd.read_pickle('LSTM Models/Univariate Optimisation/individual_error_summary.pkl')
# herd_error_df = pd.read_pickle('LSTM Models/Univariate Optimisation/herd_error_summary.pkl')
# # ensure df is sorted so plots look nice
# ind_error_df = ind_error_df.sort_values(by=['batch size', 'epochs'])
# herd_error_df = herd_error_df.sort_values(by=['batch size', 'epochs'])
# plot errors
# plot_model_error(ind_error_df, [50, 100, 150, 200, 250, 300], ['mean hourly RMSE'], "Univariate - Inidividual")
# plot_model_error(herd_error_df, [50, 100, 150, 200, 250, 300], ['mean hourly RMSE'], "Univariate - Herd")
# plot_model_error(ind_error_df, [50], ['mean hourly RMSE', 'daily freq RMSE', 'thresh_count_RMSE', 'thresh sensitivity', 'thresh specificity', 'max RMSE', 'max sensitivity', 'max specificity'], "Univariate - Inidividual")
# plot_model_error(herd_error_df, [50], ['mean hourly RMSE', 'daily freq RMSE', 'thresh_count_RMSE', 'thresh sensitivity', 'thresh specificity', 'max RMSE', 'max sensitivity', 'max specificity'], "Univariate - Herd")

########################## TESTING OF NO WEIGHTS MODEL ####################################

# test_individual_model('LSTM Models/No Weight Tests/no_weights_model.hdf5', 'Deep Learning Data/Univariate Lag 120')
# test_individual_model('LSTM Models/Multivariate Optimisation/batch_size256-100.hdf5', 'Deep Learning Data/Multivariate Lag 120')

########################## PLOT OF SUB-HERD TRENDS ####################################

# plot_subherd_trends('LSTM Models/Multivariate Optimisation/batch_size256-100.hdf5', 'Deep Learning Data/Multivariate Lag 120')

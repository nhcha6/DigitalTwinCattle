import pandas as pd
import numpy as np

def create_test_train_data_o(input_data, cows, all_data, lags, horizon, test_train):
    # define test or train
    if test_train == 'train':
        iter = range(201,1100)
    else:
        iter = range(1100,1704)

    # build test/train data
    X = []
    Y = []
    # iterate through each cow
    for cow in cows:
        # skip herd data
        if cow == 'All':
            continue
        # iterate through each sample
        for sample in iter:
            print(sample)
            # extract panting data
            panting_df = all_data["panting"]
            panting_df = panting_df[panting_df["Cow"] == cow]
            # add panting to x and y sample
            panting_df = panting_df[panting_df["next_ts"]==sample]
            x_sample = [[x] for x in panting_df[[i for i in range(-lags,0)]].values[0]]
            y_sample = [[y] for y in panting_df[[i for i in range(0,horizon)]].values[0]]
            # loop through other input data
            for input in input_data:
                # update for herd if input selected
                if input == 'herd':
                    herd_df = all_data["panting"]
                    herd_df = herd_df[herd_df["Cow"] == "All"]
                    herd_df = herd_df[herd_df["next_ts"] == sample]
                    x_new_feature = [x for x in herd_df[[i for i in range(-lags,0)]].values[0]]
                    for i in range(0,lags):
                        x_sample[i].append(x_new_feature[i])
                # update for weather
                elif input == 'HLI' or input == 'THI':
                    weather_df = all_data['weather']
                    weather_df = weather_df[input]
                    x_new_feature = weather_df.iloc[sample-lags-1:sample-1].values
                    for i in range(0,lags):
                        x_sample[i].append(x_new_feature[i])
                # update for other activity states
                else:
                    state_df = all_data[input]
                    state_df = state_df[state_df["Cow"] == cow]
                    state_df = state_df[state_df["next_ts"] == sample]
                    x_new_feature = [x for x in state_df[[i for i in range(-lags, 0)]].values[0]]
                    for i in range(0,lags):
                        x_sample[i].append(x_new_feature[i])

            X.append(x_sample)
            Y.append(y_sample)
            break
        X = np.array(X)
        Y = np.array(Y)
        break
    print(X)
    print(Y)


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
    for cow in cows:
        # skip herd data
        if cow == 'All':
            continue

        # extract panting data
        panting_df = all_data["panting"]
        panting_df = panting_df[panting_df["Cow"] == cow]

        resting_df = all_data["resting"]
        resting_df = panting_df[resting_df["Cow"] == cow]

        medium_activity_df = all_data["medium activity"]
        medium_activity_df = panting_df[medium_activity_df["Cow"] == cow]

        herd_df = all_data["panting"]
        herd_df = herd_df[herd_df["Cow"] == "All"]

        weather_df = all_data['weather']
        weather_df = weather_df['HLI']

        # iterate through each sample
        for sample in iter:
            print(sample)
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
            print(x_sample)
            a = x_sample.shape[1]
            b = x_sample.shape[0]
            x_sample = x_sample.reshape(a*b)
            x_sample = x_sample.reshape(a, b, order = 'F')

            # append
            X.append(x_sample)
            Y.append(y_sample)
            break
        X = np.array(X)
        Y = np.array(Y)
        # X = X.reshape(X.shape[0], X.shape[2], X.shape[1], order='F')
        break
    print(X)
    print(Y)

# import state behaviour data
panting_model_df = pd.read_pickle("Model Data/panting model data.pkl")
resting_model_df = pd.read_pickle("Model Data/resting model data.pkl")
medium_activity_model_df = pd.read_pickle("Model Data/medium activity model data.pkl")

# import weather data
weather_df = pd.read_csv("Weather.csv", encoding='utf-8')
weather_df = weather_df.iloc[[i for i in range(288,9163,6)]+[j for j in range(9164,10791,6)]]

# data dict
all_data_dict = {"panting": panting_model_df, "resting": resting_model_df, "medium activity": medium_activity_model_df, "weather": weather_df}

# select input data
model = ["resting", "medium activity", "HLI", "herd"]

# get cow list
cow_list = list(set(panting_model_df["Cow"]))
cow_list = sorted(cow_list)

create_test_train_data(model, cow_list, all_data_dict, 6, 24, 'train')
create_test_train_data_o(model, cow_list, all_data_dict, 6, 24, 'train')
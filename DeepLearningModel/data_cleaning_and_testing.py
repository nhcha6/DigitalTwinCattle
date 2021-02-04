from LSTM_encoder_decoder import *
import random

def analyse_test_data():
    lag=120

    # read in data
    x_train, y_train, x_test, y_test, scalar_y = read_pickle('normalised complete')

    # remove first animal
    x_train = x_train[899:]
    y_train = y_train[899:]
    x_test = x_test[604:]
    y_test = y_test[604:]

    if lag != 200:
        x_train, x_test = edit_num_lags(x_train, x_test, lag)

    x_train, y_train, x_test, y_test = add_forecast_input(x_train, y_train, x_test, y_test, 197)

    sample_weights = calculate_sample_weights_new(y_train, 7, scalar_y, False)

    print(len(x_train[0]))
    print(len(x_train[1]))

    random.seed()
    for i in range(10):

        # select random sample
        cow_no = random.randint(0,len(cleaned_cows)-1)
        print(cow_no)
        time_step = random.randint(0,int(len(x_train[0])/len(cleaned_cows)))
        print(time_step)
        sample = int(cow_no*len(x_train[0])/len(cleaned_cows) + time_step)
        print(sample)

        cow = cleaned_cows[cow_no]

        fig, ax = plt.subplots()
        ax.plot([time[0] for time in x_train[0][sample]])
        ax.set_ylabel("norm panting", fontsize=14)
        ax2 = ax.twinx()
        ax2.plot(panting_df[(panting_df["Cow"]==cow)&(panting_df["Data Type"]=="panting filtered")].values[0][202+time_step-lag:202+time_step], 'r-')
        ax2.set_ylabel("orig panting", fontsize=14)

        fig, ax = plt.subplots()
        ax.plot([time[1] for time in x_train[0][sample]])
        ax.set_ylabel("norm herd", fontsize=14)
        ax2 = ax.twinx()
        ax2.plot(panting_df[(panting_df["Cow"] == 'All') & (panting_df["Data Type"] == "panting filtered")].values[0][202 + time_step - lag:202 + time_step], 'r-')
        ax2.set_ylabel("orig herd", fontsize=14)

        fig, ax = plt.subplots()
        ax.plot([time[4] for time in x_train[0][sample]])
        ax.set_ylabel("norm medium activity", fontsize=14)
        ax2 = ax.twinx()
        ax2.plot(medium_activity_df[(medium_activity_df["Cow"] == cow) & (medium_activity_df["Data Type"] == "medium activity filtered")].values[0][202 + time_step - lag:202 + time_step], 'r-')
        ax2.set_ylabel("orig medium activity", fontsize=14)

        fig, ax = plt.subplots()
        ax.plot([time[3] for time in x_train[0][sample]])
        ax.set_ylabel("resting", fontsize=14)
        ax2 = ax.twinx()
        ax2.plot(resting_df[(resting_df["Cow"] == cow) & (resting_df["Data Type"] == "resting filtered")].values[0][202 + time_step - lag:202 + time_step], 'r-')
        ax2.set_ylabel("orig resting", fontsize=14)

        fig, ax = plt.subplots()
        ax.plot([time[2] for time in x_train[0][sample]])
        ax.set_ylabel("weather norm", fontsize=14)
        ax2 = ax.twinx()
        ax2.plot(weather_df.iloc[time_step+200-lag:time_step+200].values, 'r-')
        ax2.set_ylabel("orig weather")

        fig, ax = plt.subplots()
        ax.plot(x_train[1][sample])
        ax.set_ylabel("weather forecast", fontsize=14)
        ax2 = ax.twinx()
        ax2.plot(weather_df.iloc[time_step + 200:time_step+224].values, 'r-')
        ax2.set_ylabel("orig weather forecast")

        fig, ax = plt.subplots()
        ax.plot(y_train[sample])
        ax.set_ylabel("panting forecast", fontsize=14)
        ax2 = ax.twinx()
        ax2.plot(panting_df[(panting_df["Cow"]==cow)&(panting_df["Data Type"]=="panting filtered")].values[0][202+time_step:202+time_step+24], 'r-')
        ax2.set_ylabel("orig panting forecast")

        print(sample_weights[sample])

        plt.show()

cleaned_cows = ['8022015', '8022034', '8022073', '8022092', '8022445', '8026043', '8026045', '8026047', '8026066', '8026106', '8026154', '8026216', '8026243', '8026280', '8026304', '8026319', '8026428', '8026499', '8026522', '8026581', '8026585', '8026620', '8026621', '8026646', '8026668', '8026672', '8026699', '8026873', '8026891', '8026915', '8026953', '8026962', '8026968', '8027009', '8027035', '8027091', '8027097', '8027107', '8027181', '8027184', '8027186', '8027187', '8027207', '8027351', '8027462', '8027464', '8027476', '8027551', '8027560', '8027596', '8027603', '8027633', '8027664', '8027686', '8027688', '8027690', '8027716', '8027728', '8027752', '8027780', '8027781', '8027803', '8027808', '8027813', '8027817', '8027945', '8027951', '8028001', '8028095', '8028101', '8028105', '8028132', '8028177', '8028178', '8028186', '8028211', '8028217', '8028244', '8028255', '8028457', '8028565', '8028603', '8028649', '8028654', '8028655', '8028776', '8028811', '8028867', '8029798', '8029859', '8029865', '8029920', '8030585', '8032076', '8032104', '8032130', '8032154', '8032156', '8032169', '8032183', '8032198', '8032212', '8032229', '8032237', '8032360', '8032383', '8032468', '8032473', '8032494', '8032505', '8032506', '8032512', '8032524', '8032525', '8032526', '8032537', '8032541', '8032550', '8033173', '8033175', '8033211', '8033214', '8033215', '8033222', '8033223', '8033238', '8033246', '8033249', '8033251', '8033255', '8033275', '8033302', '8033306', '8033343', '8033348', '8033450', '8038000', '8038882', '8038884', '8038896', '8038930', '8038943', '8039058', '8039064', '8039075', '8039086', '8039093', '8039099', '8039101', '8039102', '8039116', '8039119', '8039131', '8039139', '8039143', '8039148', '8039215', '8039768', '8039813', '8039920', '8040301', '8040458', '8040459', '8040517', '8040638', '8040828', '8041081', '8042471', '8044725', '8044738', '8044842', '8045166', '8045218', '8045228', '8045535', '8045629', '8045770', '8045813', '8045831', '8045858', '8045911', '8045942', '8046335', '8046353', '8046559', '8046592', '8046782', '8047001', '8047033', '8047122', '8047228', '8047389', '8047412', '8047516', '8047842', '8047983', '8048118']
panting_df = pd.read_csv("Clean IIR Output/panting_timeseries.csv")
resting_df = pd.read_csv("Clean IIR Output/resting_timeseries.csv")
medium_activity_df = pd.read_csv("Clean IIR Output/medium activity_timeseries.csv")
weather_df = pd.read_csv("Weather.csv", encoding='utf-8')
weather_df = weather_df.iloc[[i for i in range(288,9163,6)]+[j for j in range(9164,10791,6)]]
weather_df = weather_df['HLI']

analyse_test_data()

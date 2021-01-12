# Script is used to generate plots of a selection of animals animals across either a single day or across
# a longer period of consecutive days. It does not average the behaviour across time like the hot_day_trends.py script,
# instead returning raw animal and herd data.

import matplotlib.pyplot as plt
from datetime import datetime
from average_24hr import *

# directary of animal state data and anaimal information
data_dir = '/Users/nicolaschapman/Documents/DigitalTwinCattle/DigitalTwinCattle/TrendsAndFiltering/HeatData/csvs_brisbane_18_10_18__2_1_19_tags/'
file_name = 'DailyCowStates_Brisbane Valley Feedlot_'
animal_info_name = 'Cow_ID.csv'
clean_data_dir = "Oct18-Jan1 Cleaned/AllInOne_2018Oct18to2019Jan01.csv"

# definitions of state
state_data = {0: "side lying",
              1: "resting",
              2: "medium activity",
              3: "high activity",
              4: "rumination",
              5: "eating",
              6: "walking",
              7: "grazing",
              8: "panting",
              9: "unsure",
              15: "unclassified"}

# default setting of these variables
# plot_cows = None
# cow_category = "All"
#
# # store cow info in dataframe
# cow_details_df = pd.read_csv(data_dir + animal_info_name)
# # sort cow IDs into categories using dictionaries
# breed_dict = create_category_dict("Breed", "Tag#", cow_details_df)
# coat_dict = create_category_dict("Coat colour", "Tag#", cow_details_df)
# docility_dict = create_category_dict("Docility score", "Tag#", cow_details_df)
# # select cows to be plotted, cow_category is the descriptor for the plot title.
# # cow_category = "Black"
# # plot_cows = [str(x) for x in coat_dict["Black"]]
# #plot_cows = [str(x) for x in coat_dict["White"] if str(x) in plot_cows_temp]
#
# # or we can manually select cattle
# cow_category = "8022092 - Red, 39%, F"
# plot_cows = ['8022092']

# # create date range for extracting data from excel
# total_date_list = pd.date_range(datetime(2018, 10, 19), periods=75).tolist()
# # dates of heat taken from paper
# date_set = total_date_list[16:20]
# # plot consecutive days over extended period of time
# plot_consecutive = True

# create date range for extracting data from excel
total_date_list = pd.date_range(datetime(2018, 10, 19), periods=75).tolist()

def single_day_trends(plot_cows, cow_category, state_indeces, date_set, plot_consecutive, fill_flag):
    # store all data
    all_data = {}
    for date in total_date_list:
        # upload daily data
        date_str = date.strftime("%d-%b-%Y")
        date_df = pd.read_csv(data_dir + file_name + date_str + '.csv')
        if plot_cows:
            new_columns = ["Cow"] + [x for x in plot_cows if x in date_df.columns]
            date_df = date_df[new_columns]
        all_data[date_str] = date_df

    all_data = convert_UTC_AEST(all_data)

    # convert to fill data if requested
    if fill_flag:
        all_data = convert_to_fill(all_data)

    print(all_data)

    # isolate days to plot
    daily_data = {}
    # import data from excel for each date
    for date_str, data in all_data.items():
        if date_str in date_set:
            # add to date_df
            daily_data[date_str] = data

    # define x-axis
    x_axis = list(range(1,25))

    consecutive_data = {}
    for date_str in date_set:
        date_df = daily_data[date_str]
        # create plots for each state
        for state_index in state_indeces:
            # plotting feedback
            print("Plotting " + str(state_data[state_index]) + " for " + date_str)
            # calculate the average day for each dataset
            state_day = create_mins_per_hour(date_df, state_index)
            state_day = average_cows(state_day)

            if plot_consecutive:
                if state_index in consecutive_data.keys():
                    consecutive_data[state_index] += state_day
                else:
                    consecutive_data[state_index] = state_day
            else:
                # plot the daily data set
                plt.figure()
                plt.plot(x_axis,state_day)
                plt.title("Time spent " + str(state_data[state_index]) + " " + date_str + " (" + cow_category + ")", fontsize=10)
                plt.xlabel("Hour of the Day")
                plt.ylabel("Minutes " + str(state_data[state_index]) + " per hour")

    if plot_consecutive:
        for state_index, data in consecutive_data.items():
            plt.figure()
            plt.plot(data)
            plt.title("Time spent " + str(state_data[state_index]) + " " + date_set[0] + " - " + date_set[-1] + " (" + cow_category + ")",fontsize=10)
            plt.xlabel("Hour of the Day")
            plt.ylabel("Minutes " + str(state_data[state_index]) + " per hour")

        # return the most recent data point for use in filtering
        return data

    # return most recent state_day if not running consecutive
    return state_day

def single_day_trends_clean(plot_cows, cow_category, state_indeces, date_set, plot_consecutive, fill_flag):
    # # store all data
    # all_data = {}
    # for date in total_date_list:
    #     # upload daily data
    #     date_str = date.strftime("%d-%b-%Y")
    #     date_df = pd.read_csv(data_dir + file_name + date_str + '.csv')
    #     if plot_cows:
    #         new_columns = ["Cow"] + [x for x in plot_cows if x in date_df.columns]
    #         date_df = date_df[new_columns]
    #     all_data[date_str] = date_df
    #
    # all_data = convert_UTC_AEST(all_data)

    # import clean data
    cleaned_data_df = pd.read_csv(clean_data_dir)
    # select only relevant cows
    new_columns = ["AEST_Date", "AEST_Time"] + [x for x in cleaned_data_df.columns if x[1:] in plot_cows]
    cleaned_data_df = cleaned_data_df[new_columns]
    print(cleaned_data_df)

    # convert to fill data if requested
    if fill_flag:
        cleaned_data_df = convert_cleaned_to_fill(cleaned_data_df)

    # # isolate days to plot
    # daily_data = {}
    # # import data from excel for each date
    # for date_str, data in all_data.items():
    #     if date_str in date_set:
    #         # add to date_df
    #         daily_data[date_str] = data

    daily_data = {}
    for date_str in date_set:
        date_time_obj = datetime.strptime(date_str, "%d-%b-%Y")
        new_date_str = date_time_obj.strftime("%Y-%m-%d")
        print(new_date_str)

        daily_data[date_str] = cleaned_data_df[cleaned_data_df["AEST_Date"]==new_date_str]

    print(daily_data)

    # define x-axis
    x_axis = list(range(1, 25))

    consecutive_data = {}
    for date_str in date_set:
        date_df = daily_data[date_str]
        # create plots for each state
        for state_index in state_indeces:
            # plotting feedback
            print("Plotting " + str(state_data[state_index]) + " for " + date_str)
            # calculate the average day for each dataset
            state_day = create_mins_per_hour(date_df, state_index)
            state_day = average_cows(state_day)

            if plot_consecutive:
                if state_index in consecutive_data.keys():
                    consecutive_data[state_index] += state_day
                else:
                    consecutive_data[state_index] = state_day
            else:
                # plot the daily data set
                plt.figure()
                plt.plot(x_axis, state_day)
                plt.title("Time spent " + str(state_data[state_index]) + " " + date_str + " (" + cow_category + ")",
                          fontsize=10)
                plt.xlabel("Hour of the Day")
                plt.ylabel("Minutes " + str(state_data[state_index]) + " per hour")

    if plot_consecutive:
        for state_index, data in consecutive_data.items():
            plt.figure()
            plt.plot(data)
            plt.title("Time spent " + str(state_data[state_index]) + " " + date_set[0] + " - " + date_set[
                -1] + " (" + cow_category + ")", fontsize=10)
            plt.xlabel("Hour of the Day")
            plt.ylabel("Minutes " + str(state_data[state_index]) + " per hour")

        # return the most recent data point for use in filtering
        return data

    # return most recent state_day if not running consecutive
    return state_day
# Used to generate and compare plots of average herd behaviour (can select the entire herd or any sub-group of
# the herd using categorical cow data) over different time periods. That is, it generates and average 24 hour
# cycle of a select group of cows over select dates.

import matplotlib.pyplot as plt
from datetime import datetime
from average_24hr import *

# directary of animal state data and anaimal information
data_dir = 'HeatData/csvs_brisbane_18_10_18__2_1_19_tags/'
file_name = 'DailyCowStates_Brisbane Valley Feedlot_'
animal_info_name = 'Cow_ID.csv'

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

# #default setting of these variables
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
# #or we can manually select cattle
# # cow_category = "8027107 - White, 50%, F"
# # plot_cows = ['8027107']

# create date range for extracting data from excel
total_date_list = pd.date_range(datetime(2018, 10, 19), periods=75).tolist()
# dates of heat taken from paper
hot_date_list = total_date_list[16:20] + total_date_list[42:46] + total_date_list[62:65]
hot_date_set = set([date.strftime("%d-%b-%Y") for date in hot_date_list])

def hot_day_trends(plot_cows, cow_category, state_indeces):
    # split into hot data and other data
    heat_data = {}
    other_data = {}
    all_data = {}

    # import data from excel for each date
    for date in total_date_list:
        # upload daily data
        date_str = date.strftime("%d-%b-%Y")
        date_df = pd.read_csv(data_dir + file_name + date_str + '.csv')
        if plot_cows:
            new_columns = ["Cow"] + [x for x in plot_cows if x in date_df.columns]
            date_df = date_df[new_columns]
        all_data[date_str] = date_df

    all_data_AEST = convert_UTC_AEST(all_data)

    for date_str, date_df in all_data_AEST.items():
        if date_str in hot_date_set:
            heat_data[date_str] = date_df
        else:
            other_data[date_str] = date_df

    print(heat_data)

    # define states to run and default x_axis
    x_axis = list(range(1,25))

    # create plots for each state
    for state_index in state_indeces:
        print("Plotting for " + str(state_data[state_index]))
        # calculate the average day for each dataset
        ave_day_other = average_day(other_data, state_index)
        print("First dataset complete")
        ave_day_heat = average_day(heat_data, state_index)
        print("Second dataset complete")

        # plot the two data sets
        plt.figure()
        plt.plot(x_axis,ave_day_other, label="other")
        plt.plot(x_axis,ave_day_heat, label="hot")
        plt.legend(loc="upper right")
        plt.title("Average time spent " + str(state_data[state_index]) + " (" + cow_category + " 19-Oct-18 to 1-Jan-19)", fontsize=10)
        plt.xlabel("Hour of the Day")
        plt.ylabel("Average minutes " + str(state_data[state_index]) + " per hour")

    # return the data of the most recent state run
    return ave_day_heat, ave_day_other


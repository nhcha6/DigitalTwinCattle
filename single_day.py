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
              15: "invalid data"}

# default setting of these variables
plot_cows = None
cow_category = "All"

# store cow info in dataframe
cow_details_df = pd.read_csv(data_dir + animal_info_name)
# sort cow IDs into categories using dictionaries
breed_dict = create_category_dict("Breed", "Tag#", cow_details_df)
coat_dict = create_category_dict("Coat colour", "Tag#", cow_details_df)
docility_dict = create_category_dict("Docility score", "Tag#", cow_details_df)
# select cows to be plotted, cow_category is the descriptor for the plot title.
# cow_category = "Black"
# plot_cows = [str(x) for x in coat_dict["Black"]]
#plot_cows = [str(x) for x in coat_dict["White"] if str(x) in plot_cows_temp]

# or we can manually select cattle
cow_category = "8022092 - Red, 39%, F"
plot_cows = ['8022092']

# create date range for extracting data from excel
total_date_list = pd.date_range(datetime(2018, 10, 19), periods=75).tolist()
# dates of heat taken from paper
date_set = total_date_list[50:54]
# plot consecutive days over extended period of time
plot_consecutive = False

# split into hot data and other data
daily_data = {}

# import data from excel for each date
for date in date_set:
    # upload daily data
    date_str = date.strftime("%d-%b-%Y")
    date_df = pd.read_csv(data_dir + file_name + date_str + '.csv')
    # update for individual cows only
    if plot_cows:
        new_columns = ["Cow"] + [x for x in plot_cows if x in date_df.columns]
        date_df = date_df[new_columns]
    # add to date_df
    daily_data[date_str] = date_df

# define states to run and default x_axis
# state_indeces = [1, 2, 3, 4, 5, 6, 7, 8, 9, 15]
state_indeces = [1, 5, 8]
x_axis = list(range(1,25))

i = 0
consecutive_data = {}
for date in date_set:
    date_str = date.strftime("%d-%b-%Y")
    date_df = daily_data[date_str]
    # create plots for each state
    for state_index in state_indeces:
        # figure number
        i+=1
        # plotting feedback
        print("Plotting " + str(state_data[state_index]) + " for " + date_str)
        # calculate the average day for each dataset
        state_day = create_mins_per_hour(date_df, state_index)
        state_day = state_day.iloc[0,1:].tolist()
        # rotate list to transfer times to GMT+10
        state_day = state_day[-10:] + state_day[:-10]

        if plot_consecutive:
            if state_index in consecutive_data.keys():
                consecutive_data[state_index] += state_day
            else:
                consecutive_data[state_index] = state_day
        else:
            # plot the daily data set
            plt.figure(i)
            plt.plot(x_axis,state_day)
            plt.title("Time spent " + str(state_data[state_index]) + " " + date_str + " (" + cow_category + ")", fontsize=10)
            plt.xlabel("Hour of the Day")
            plt.ylabel("Minutes " + str(state_data[state_index]) + " per hour")

if plot_consecutive:
    for state_index, data in consecutive_data.items():
        plt.figure(state_index)
        plt.plot(data)
        plt.title("Time spent " + str(state_data[state_index]) + " " + date_set[0].strftime("%d-%b-%Y") + " - " + date_set[-1].strftime("%d-%b-%Y") + " (" + cow_category + ")",fontsize=10)
        plt.xlabel("Hour of the Day")
        plt.ylabel("Minutes " + str(state_data[state_index]) + " per hour")



# show plots
plt.show()
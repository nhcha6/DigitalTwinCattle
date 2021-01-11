import sys
sys.path.insert(1, '/Users/nicolaschapman/Documents/DigitalTwinCattle/DigitalTwinCattle/TrendsAndFiltering')

from single_day import *
from hot_day_trends import *
from filter_data import *
from area_under_curve import *

# df = pd.read_csv("panting_regression.csv")
# print(df)

###################### ESTABLISH REQUIRED DATA ##################

# default setting of these variables
plot_cows = None
cow_category = "All"

# store cow info in dataframe
cow_details_df = pd.read_csv(data_dir + animal_info_name)

# sort cow IDs into categories using dictionaries
breed_dict = create_category_dict("Breed", "Tag#", cow_details_df)
coat_dict = create_category_dict("Coat colour", "Tag#", cow_details_df)
docility_dict = create_category_dict("Docility score", "Tag#", cow_details_df)

#################################################################

################# SELECT COWS TO BE PLOTTED #####################

# select cows to be plotted, cow_category is the descriptor for the plot title.
# cow_category = "Black"
# plot_cows = [str(x) for x in coat_dict["Black"]]

# or we can manually select cattle
#cow_category = "8027107 - White, 50%, F"
#plot_cows = ['8027107']
# cow_category = "8022092 - Red, 39%, F"
# plot_cows = ['8022092']

# list of cows for regression analysis
#cows = ['8027107', '8022092', '8027476', '8032505', '8045911']
cows = cow_details_df["Tag#"]
#cows = ['8026433']

##################################################################

############## SELECT DATES AND STATE TO ANALYSE #################

# can also select the specific dates we wish to plot via single_day()
total_date_list = pd.date_range(datetime(2018, 10, 19), periods=75).tolist()
# all dates to be plotted consecutively for all time series data
date_set = [date.strftime("%d-%b-%Y") for date in total_date_list[1:-1]]
# plot consecutive days over extended period of time
plot_consecutive = True

# select state indeces
state_indeces = [8]

##################################################################

############## GENERATE ALL DATA FOR REGRESSION ANALYSIS #################

excel_data = []
for index in state_indeces:
    i = 0
    for plot_cow in cows:
        i+=1
        print(i, plot_cow)
        plot_cow = str(plot_cow)
        # Extract raw data for current cow
        cow_data = [state_data[state_indeces[0]]+" raw"]
        cow_data.append(plot_cow)
        signal = single_day_trends([plot_cow], plot_cow, state_indeces, date_set, plot_consecutive, True)
        cow_data += signal
        excel_data.append(cow_data)

        # IIR LP filter
        cow_data = [state_data[state_indeces[0]] + " filtered"]
        cow_data.append(plot_cow)
        filtered_signal = butter_lp_filter([3], [4], signal, plot_cow)
        cow_data += list(filtered_signal)
        excel_data.append(cow_data)

    # extract herd behaviour
    cow_data = [state_data[state_indeces[0]] + " raw"]
    cow_data.append("All")
    all_cows = single_day_trends(None, "All", state_indeces, date_set, plot_consecutive, True)
    cow_data+=all_cows
    excel_data.append(cow_data)
    # filtered herd behaviour
    cow_data = [state_data[state_indeces[0]] + " filtered"]
    cow_data.append("All")
    filtered_herd = butter_lp_filter([3], [4], all_cows, 'All')
    cow_data += list(filtered_herd)
    excel_data.append(cow_data)

column_headers = ['Data Type', 'Cow']
column_headers += [x for x in range(1,1753)]
regression_df = pd.DataFrame(excel_data, columns=column_headers)
regression_df.to_csv("Entire Dataset Output/" + state_data[state_indeces[0]] + "_regression.csv", index = False)

####################################################

################ AREA UNDER GRAPH ##################

# area_signal = area_under_graph(filtered_signal)
# print(area_signal)
# area_herd = area_under_graph(filtered_herd)
# print(area_herd)
# area_animal = area_under_graph(ave_day_other)
# print(area_animal)

# herd_comp, animal_ave_comp = heat_stress_comp(filtered_signal, filtered_herd, ave_day_other)
# inverse when running for resting!
# herd_comp = [1/x for x in herd_comp]
# animal_ave_comp = [1/x for x in animal_ave_comp]
# print(herd_comp)
# print(animal_ave_comp)

####################################################

# show plots
plt.show()

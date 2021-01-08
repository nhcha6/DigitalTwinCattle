from single_day import *
from hot_day_trends import *
from filter_data import *
from clustering import *

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
cow_category = "8027107 - White, 50%, F"
plot_cows = ['8027107']
# cow_category = "8022092 - Red, 39%, F"
# plot_cows = ['8022092']

##################################################################

############## SELECT DATES AND STATE TO ANALYSE #################

# can also select the specific dates we wish to plot via single_day()
total_date_list = pd.date_range(datetime(2018, 10, 19), periods=75).tolist()
# dates of heat taken from paper
date_set = [date.strftime("%d-%b-%Y") for date in total_date_list[16:20]]
#date_set = [date.strftime("%d-%b-%Y") for date in total_date_list[32:36]]
#date_set = [date.strftime("%d-%b-%Y") for date in total_date_list[32:42]]
# plot consecutive days over extended period of time
plot_consecutive = True

# select state indeces
state_indeces = [4]

##################################################################

########### HOT VS NON-HOT DAY TRENDS #############

ave_day_heat, ave_day_other = hot_day_trends(plot_cows, cow_category, state_indeces, True)
# extrapolate to length of signal
#hot_days_ex, other_days_ex = extrapolate_heat(int(len(signal)/24), ave_day_heat, ave_day_other)

####################################################


############### SINGLE DAY ANALYSIS ################

signal = single_day_trends(plot_cows, cow_category, state_indeces, date_set, plot_consecutive, True)
hot_days = single_day_trends(None, "All", state_indeces, date_set, plot_consecutive, True)
#other_days = single_day_trends(None, "All", state_indeces, total_date_list[24:28], plot_consecutive)

####################################################

################ FILTERING ANALYSIS ##################

# WAVELET
# DWT_level_2(hot_days, "hot", other_days, "other", signal, "signal")

# FFT
#fourier_transform(signal, cow_category)
#fourier_transform(hot_days, "All Animals")
#fourier_transform(hot_days_ex, "Hot Day Average")

# FIR LP
# cutoffs = [i+0.5 for i in range(1,6,2)]
# widths = [4,8]
# fil_lp_filter(cutoffs, widths, signal, cow_category)
# filtered_signal = fil_lp_filter([3], [7], signal, cow_category)
# fil_lp_filter([4], [7], hot_days, 'All')

# IIR LP
filtered_signal = butter_lp_filter([4], [4], signal, cow_category)
filtered_herd = butter_lp_filter([4], [4], hot_days, 'All')

####################################################

################ AREA UNDER GRAPH ##################

# area_signal = area_under_graph(filtered_signal)
# print(area_signal)
# area_herd = area_under_graph(filtered_herd)
# print(area_herd)
# area_animal = area_under_graph(ave_day_other)
# print(area_animal)

herd_comp, animal_ave_comp = heat_stress_comp(filtered_signal, filtered_herd, ave_day_other)
# inverse when running for resting!
# herd_comp = [1/x for x in herd_comp]
# animal_ave_comp = [1/x for x in animal_ave_comp]
print(herd_comp)
print(animal_ave_comp)

####################################################

# show plots
plt.show()

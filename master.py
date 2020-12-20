from single_day import *
from hot_day_trends import *
from filter_data import *

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

# or we can manually select cattle
# cow_category = "8027107 - White, 50%, F"
# plot_cows = ['8027107']
cow_category = "8022092 - Red, 39%, F"
plot_cows = ['8022092']

# can also select the specific dates we wish to plot via single_day()
total_date_list = pd.date_range(datetime(2018, 10, 19), periods=75).tolist()
# dates of heat taken from paper
date_set = [date.strftime("%d-%b-%Y") for date in total_date_list[16:20]]
#date_set = total_date_list[42:46]
# plot consecutive days over extended period of time
plot_consecutive = True

# select state indeces
state_indeces = [8]

# run hot_day_trends
#ave_day_heat, ave_day_other = hot_day_trends(None, "All", state_indeces)


# run single_day()
signal = single_day_trends(plot_cows, cow_category, state_indeces, date_set, plot_consecutive)
hot_days = single_day_trends(None, "All", state_indeces, date_set, plot_consecutive)
#other_days = single_day_trends(None, "All", state_indeces, total_date_list[24:28], plot_consecutive)

# extrapolate to length of signal
#hot_days_ex, other_days_ex = extrapolate_heat(int(len(signal)/24), ave_day_heat, ave_day_other)

# run a filtering algorithm and plot
# DWT_level_2(hot_days, "hot", other_days, "other", signal, "signal")

# run fft
fourier_transform(signal, cow_category)
fourier_transform(hot_days, "All Animals")
#fourier_transform(hot_days_ex, "Hot Day Average")

# cutoffs = [i+0.5 for i in range(1,6,2)]
# widths = [4,8]
# fil_lp_filter(cutoffs, widths, signal, cow_category)
fil_lp_filter([3.5], [8], signal, cow_category)
fil_lp_filter([3.5], [8], hot_days, 'All')

# show plots
plt.show()
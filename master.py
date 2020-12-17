from single_day import *
from hot_day_trends import *

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
cow_category = "Black"
plot_cows = [str(x) for x in coat_dict["Black"]]

# or we can manually select cattle
# cow_category = "8027107 - White, 50%, F"
# plot_cows = ['8027107']

# can also select the specific dates we wish to plot via single_day()
total_date_list = pd.date_range(datetime(2018, 10, 19), periods=75).tolist()
# dates of heat taken from paper
date_set = total_date_list[16:18]
# plot consecutive days over extended period of time
plot_consecutive = False

# select state indeces
state_indeces = [5]

# run hot_day_trends
hot_day_trends(plot_cows, cow_category, state_indeces)

# run single_day()
single_day_trends(plot_cows, cow_category, state_indeces, date_set, plot_consecutive)

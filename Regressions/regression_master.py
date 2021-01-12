import sys
sys.path.insert(1, '/Users/nicolaschapman/Documents/DigitalTwinCattle/DigitalTwinCattle/TrendsAndFiltering')

from single_day import *
from hot_day_trends import *
from filter_data import *
from area_under_curve import *

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

# all cleaned cows
# clean_data_dir = "/Users/nicolaschapman/Documents/DigitalTwinCattle/DigitalTwinCattle/TrendsAndFiltering/Oct18-Jan1 Cleaned/AllInOne_2018Oct18to2019Jan01.csv"
# cleaned_df = pd.read_csv(clean_data_dir)
# cleaned_cows = cleaned_df.columns[8:]
# cleaned_cows = [x[1:] for x in cleaned_cows]
cleaned_cows = ['8021870', '8022015', '8022034', '8022073', '8022092', '8022445', '8026043', '8026045', '8026047', '8026066', '8026106', '8026154', '8026216', '8026243', '8026280', '8026304', '8026319', '8026428', '8026499', '8026522', '8026581', '8026585', '8026620', '8026621', '8026646', '8026668', '8026672', '8026699', '8026873', '8026891', '8026915', '8026953', '8026962', '8026968', '8027009', '8027035', '8027091', '8027097', '8027107', '8027181', '8027184', '8027186', '8027187', '8027207', '8027351', '8027462', '8027464', '8027476', '8027551', '8027560', '8027596', '8027603', '8027633', '8027664', '8027686', '8027688', '8027690', '8027716', '8027728', '8027752', '8027780', '8027781', '8027803', '8027808', '8027813', '8027817', '8027945', '8027951', '8028001', '8028095', '8028101', '8028105', '8028132', '8028177', '8028178', '8028186', '8028211', '8028217', '8028244', '8028255', '8028457', '8028565', '8028603', '8028649', '8028654', '8028655', '8028776', '8028811', '8028867', '8029798', '8029859', '8029865', '8029920', '8030585', '8032076', '8032104', '8032130', '8032154', '8032156', '8032169', '8032183', '8032198', '8032212', '8032229', '8032237', '8032360', '8032383', '8032468', '8032473', '8032494', '8032505', '8032506', '8032512', '8032524', '8032525', '8032526', '8032537', '8032541', '8032550', '8033173', '8033175', '8033211', '8033214', '8033215', '8033222', '8033223', '8033238', '8033246', '8033249', '8033251', '8033255', '8033275', '8033302', '8033306', '8033343', '8033348', '8033450', '8038000', '8038882', '8038884', '8038896', '8038930', '8038943', '8039058', '8039064', '8039075', '8039086', '8039093', '8039099', '8039101', '8039102', '8039116', '8039119', '8039131', '8039139', '8039143', '8039148', '8039215', '8039768', '8039813', '8039920', '8040301', '8040458', '8040459', '8040517', '8040638', '8040828', '8041081', '8042471', '8044725', '8044738', '8044842', '8045166', '8045218', '8045228', '8045535', '8045629', '8045770', '8045813', '8045831', '8045858', '8045911', '8045942', '8046335', '8046353', '8046559', '8046592', '8046782', '8047001', '8047033', '8047122', '8047228', '8047389', '8047412', '8047516', '8047842', '8047983', '8048118']

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
#cows = ['8026433']
cows = cleaned_cows

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
cutoff = 3

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
        signal = single_day_trends_clean([plot_cow], plot_cow, state_indeces, date_set, plot_consecutive, True)
        cow_data += signal
        excel_data.append(cow_data)

        # IIR LP filter
        cow_data = [state_data[state_indeces[0]] + " filtered"]
        cow_data.append(plot_cow)
        filtered_signal = butter_lp_filter([cutoff], [4], signal, plot_cow)
        cow_data += list(filtered_signal)
        excel_data.append(cow_data)

    # extract herd behaviour
    cow_data = [state_data[state_indeces[0]] + " raw"]
    cow_data.append("All")
    all_cows = single_day_trends_clean(None, "All", state_indeces, date_set, plot_consecutive, True)
    cow_data+=all_cows
    excel_data.append(cow_data)
    # filtered herd behaviour
    cow_data = [state_data[state_indeces[0]] + " filtered"]
    cow_data.append("All")
    filtered_herd = butter_lp_filter([cutoff], [4], all_cows, 'All')
    cow_data += list(filtered_herd)
    excel_data.append(cow_data)

column_headers = ['Data Type', 'Cow']
column_headers += [x for x in range(1,1753)]
regression_df = pd.DataFrame(excel_data, columns=column_headers)
regression_df.to_csv("Clean Dataset Output/" + state_data[state_indeces[0]] + "_timeseries.csv", index = False)

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
#plt.show()

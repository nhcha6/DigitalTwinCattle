# Script contains functions for converting cattle behaviour state data contained in excel files
# to summary data which represents the average 24 hour cycle of the herd for given behaviour types.
# see average_day function for details of inputs.

import pandas as pd
#pd.set_option("display.max_rows", None, "display.max_columns", None)

def average_day(data, state):
    """
    takes the raw excel data and converts it to the average 24 hours cycle of the herd for the
    selected state.

    :param data: a dictionary containing dates as keys and the corresponding excel data stored in
    a dataframe as the values.
    :param state: a number of range 1-9 or 15 denoting the animal state of interest.
    :return:
    """

    average_day = []
    # loop through the raw data from each day
    for key, value in data.items():
        if not len(value.index) == 1440:
            continue
        # calculate minute spent in given state per hour for each animal
        per_hour_df = create_mins_per_hour(value, state)
        # average mins per hour for a given day across all animals
        average_mins_day = average_cows(per_hour_df)
        # create a list of the distributions for each day
        for i in range(0,len(average_mins_day)):
            if not len(average_day)==24:
                average_day.append([average_mins_day[i]])
            else:
                average_day[i].append(average_mins_day[i])
    # find average of distribution
    for i in range(0, len(average_day)):
        average_day[i] = sum(average_day[i])/len(average_day[i])
    return average_day

def create_mins_per_hour(df, state):
    """
    takes the dataframe of single day with per minute cow state recorded for every cow. This function
    converts this to a dataframe where each row corresponds to a cow and each column corresponds to an
    hour of the day. The time per hour spent in the input state is calculated and stored in each element.

    :param df:
    :param state:
    :return:
    """
    per_hour_list = []
    # iterate through each cow (columns)
    for cow in df.columns:
        # ignore first column as this is the time.
        if cow == 'Cow':
            continue
        # extract the cow df
        cow_df = df[cow]
        new_cow_data = [cow]
        # iterate through the row of each cow and count the time spent in the given state each hour.
        state_count = 0
        for index, value in cow_df.items():
            if value == state:
                state_count+=1
            if not (index+1)%60:
                new_cow_data.append(state_count)
                state_count=0
        # add this to the list
        per_hour_list.append(new_cow_data)

    # create new columns headings, the cow ID and the hour of the day.
    columns = ["Cow"] + list(range(0,24))
    # convert data to dataframe.
    per_hour_df = pd.DataFrame(per_hour_list, columns = columns)
    return per_hour_df


def average_cows(per_hour_df):
    """
    recieves the dataframe summarising time spent in the given state each hour for each animal across a
    given day and finds the average for each hour across all animals.

    :param per_hour_df:
    :return:
    """
    # iterate through each time, calculate the average time spent in each state and add to list.
    daily_average_mins = []
    for time in per_hour_df.columns:
        if time == 'Cow':
            continue
        average_mins = per_hour_df[time].mean()
        daily_average_mins.append(average_mins)

    return daily_average_mins

def create_category_dict(category_header, cow_ID_header, details_df):
    categories = set(details_df[category_header])
    category_dict = {key: [] for key in categories}
    for category in categories:
        category_dict[category] = details_df[cow_ID_header][details_df[category_header] == category].tolist()
    #print(category_dict)
    return category_dict

def convert_UTC_AEST(dict_UTC):
    """
    creates new identical dictionary, except with dates and time in UTC instead of AEST. Note that the
    first date included in the data is droppped as it doesn't include a full cycle.

    :param dict_UTC:
    :return:
    """
    new_dict_AEST = {}
    past_flag = False

    for key, value in dict_UTC.items():
        if past_flag:
            new_df = pd.concat([past_df, value.iloc[0:840]], axis=0)
            new_df = new_df.reset_index(drop=True)
            new_dict_AEST[key] = new_df
        past_df = value.iloc[840:]
        past_flag = True

    return new_dict_AEST

def convert_to_fill(cows_dict):
    """
    This function takes the raw cow data dictionary (as it was imported from the csv) and converts it to
    the 'fill' data. That is, if two consecutive minutes have the same state, this state is filled
    until two consecutive minutes of an additional state are recorded.

    :param cows_dict:
    :return:
    """
    # declare new dictionary
    fill_cows_dict = {}
    # previous date to ensure fill across dates
    prev_date = None
    # loop through each data of data
    for date, df in cows_dict.items():
        # deep copy of the df for the current data
        df_new = df.copy()
        # iterate through each cow
        for cow in df.columns:
            # skip first header
            if cow == 'Cow':
                continue
            # define variables for creating fill data (continuity between days maintained)
            try:
                prev_state = cows_dict[prev_date][cow].iloc[1439]
                fill_state = fill_cows_dict[prev_date][cow].iloc[1439]
            except KeyError:
                prev_state = None
                fill_state = None
            # iterate each state and build a new column of fill data
            new_col = []
            for index, state in df[cow].items():
                # update fill_state if previous two states are the same
                if state == prev_state:
                    fill_state = state
                # if fill_state is not None, we append it.
                if fill_state:
                    new_col.append(fill_state)
                else:
                    new_col.append(state)
                # update previous state
                prev_state = state
            # update new df with fill data
            df_new[cow] = new_col
        # update fill date/data dictionary with the new dict
        fill_cows_dict[date] = df_new
        prev_date = date

    return fill_cows_dict

def convert_cleaned_to_fill(cleaned_df):
    df_new = cleaned_df.copy()
    for cow in cleaned_df.columns:
        # skip time columns
        if cow=='AEST_Time' or cow=='AEST_Date':
            continue

        # update fill data
        prev_state = None
        fill_state = None
        new_col = []
        for index, state in cleaned_df[cow].items():
            # first iteration has no prev_state
            if not index:
                prev_state = state

            # else update fill_state and prev_state
            else:
                if state==prev_state:
                    fill_state = state
                prev_state = state

            # if fill_state exists, update col. Otherwise use current state.
            if fill_state:
                new_col.append(fill_state)
            else:
                new_col.append(state)

        df_new[cow] = new_col

    return df_new





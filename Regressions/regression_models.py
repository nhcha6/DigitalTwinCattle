import sys
sys.path.insert(1, '/Users/nicolaschapman/Documents/DigitalTwinCattle/DigitalTwinCattle/TrendsAndFiltering')

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from filter_data import *

def build_test_data(lower, upper, rows):
    x = []
    y = []
    for i in range(lower,upper,24):
        # print(i)
        [raw_ts, filtered_ts] = rows[[str(j) for j in range(i, i + 103)]].values.tolist()
        # x_data = time_series[0]+time_series[1]
        filtered_x = butter_lp_filter([3], [4], raw_ts, "All", False)
        x_data = [(raw_ts[j]-raw_ts[j-24]) for j in range(24,len(raw_ts))]
        x_data.extend([(filtered_x[k]-filtered_x[k-24]) for k in range(24,len(filtered_x))])
        # herd_data = herd_rows[[str(j) for j in range(i,i+31)]].values.tolist()
        # x_data = x_data + herd_data[1] + herd_data[0]

        y_data = rows.iloc[1][str(i + 104)] - rows.iloc[1][str(i + 80)]

        # train to predict area under.
        # for day in range(0,96,24):
        #     day_data = rows[[str(j) for j in range(i+day,i+day+24)]].values.tolist()
        #     x_data.append(sum(day_data[1]))
        #
        # next_day_data = rows[[str(j) for j in range(i+96,i+96+24)]].values.tolist()
        # y_data = sum(next_day_data[1])

        # if np.isnan(x_data).any():
        #     print("invalid data for " + cow)
        #     break

        x.append(x_data)
        y.append(y_data)

    return x, y

df_panting = pd.read_csv("Clean Dataset Output/panting_timeseries.csv")

# ensure iterates through data in a consistent manner
cow_list = list(set(df_panting["Cow"]))
cow_list = sorted(cow_list)
#cow_list = ['8027107', '8022092', '8027476', '8032505', '8045911']
#cow_list = ['8027107']
data_type_list = list(set(df_panting["Data Type"]))
data_type_list = sorted(data_type_list)

# extract herd data
herd_rows = df_panting.loc[(df_panting["Cow"] == "All")]

# iterate through each animal and data type to build input data
x = []
y = []
x_test = []
y_test = []
counter = 0
for cow in cow_list:
    counter+=1
    print(counter)
    print(cow)

    if cow=='All':
        continue

    # if counter==2:
    #     break

    rows = df_panting.loc[(df_panting["Cow"] == cow)]

    x_cow, y_cow = build_test_data(1, 1200, rows)
    x.extend(x_cow)
    y.extend(y_cow)

    x_test_cow, y_test_cow = build_test_data(1200, 1644, rows)
    x_test.extend(x_test_cow)
    y_test.extend(y_test_cow)

x = np.array(x)
y = np.array(y)
print(np.shape(x))
print(np.shape(y))
print(x[0])
print(y)

X = sm.add_constant(x)
results = sm.OLS(y,X).fit()

print(results.summary())

y_pred = results.predict(X)
print("In sample RMSE: " + str(mean_squared_error(y, y_pred, squared=False)))
print("In sample R-squared: " + str(r2_score(y, y_pred)))

X_test = sm.add_constant(x_test)
y_test_pred = results.predict(X_test)
print("Out of sample RMSE: " + str(mean_squared_error(y_test, y_test_pred, squared=False)))
print("Out of sample R-squared: " + str(r2_score(y_test, y_test_pred)))

coeffs = results.params
pvals = results.pvalues

fig, ax1 = plt.subplots()
ax1.plot(coeffs)
ax2 = ax1.twinx()
ax2.plot(pvals, 'ro')

fig.tight_layout()
plt.show()

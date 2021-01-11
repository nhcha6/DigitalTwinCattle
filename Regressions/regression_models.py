import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.tools.sm_exceptions import MissingDataError

df_panting = pd.read_csv("Entire Dataset Output/panting_regression.csv")

x = []
y = []

# ensure iterates through data in a consistent manner
cow_list = list(set(df_panting["Cow"]))
cow_list = sorted(cow_list)
#cow_list = ['8027107', '8022092', '8027476', '8032505', '8045911']
#cow_list = ['8027107']
data_type_list = list(set(df_panting["Data Type"]))
data_type_list = sorted(data_type_list)

# extract herd data
herd_rows = df_panting.loc[(df_panting["Cow"] == "All")]

# iterate through each animal and data type
counter = 0
for cow in cow_list:
    counter+=1
    print(counter)
    # print(cow)
    if cow=='All':
        continue
    rows = df_panting.loc[(df_panting["Cow"] == cow)]

    for i in range(1,1200, 24):
        # print(i)
        x_data = rows[[str(j) for j in range(i,i+31)]].values.tolist()
        x_data = x_data[0]+x_data[1]
        # herd_data = herd_rows[[str(j) for j in range(i,i+31)]].values.tolist()
        # x_data = x_data + herd_data[1] + herd_data[0]
        y_data = rows.iloc[1][str(i+37)]

        # train to predict area under.
        # for day in range(0,96,24):
        #     day_data = rows[[str(j) for j in range(i+day,i+day+24)]].values.tolist()
        #     x_data.append(sum(day_data[1]))
        #
        # next_day_data = rows[[str(j) for j in range(i+96,i+96+24)]].values.tolist()
        # y_data = sum(next_day_data[1])

        if np.isnan(x_data).any():
            print("invalid data for " + cow)
            break
        x.append(x_data)
        y.append(y_data)

x = np.array(x)
y = np.array(y)
print(np.shape(x))
print(np.shape(y))
print(x[0])
print(y)

X = sm.add_constant(x)
results = sm.OLS(y,X).fit()

print(results.summary())


coeffs = results.params
pvals = results.pvalues

fig, ax1 = plt.subplots()
ax1.plot(coeffs)
ax2 = ax1.twinx()
ax2.plot(pvals, 'ro')

fig.tight_layout()
plt.show()

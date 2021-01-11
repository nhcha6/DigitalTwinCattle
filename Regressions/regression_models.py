import numpy as np
import pandas as pd
import statsmodels.api as sm

df = pd.read_csv("Five Animal Output/panting_regression.csv")

x = []
y = []

# ensure iterates through data in a consistent manner
cow_list = list(set(df["Cow"]))
cow_list = sorted(cow_list)
data_type_list = list(set(df["Data Type"]))
data_type_list = sorted(data_type_list)

# iterate through each animal and data type
for cow in cow_list:
    for i in range(1,1657, 24):
        print(cow)
        print(i)
        rows = df.loc[(df["Cow"]==cow)]
        x_data = rows[[str(j) for j in range(i,i+96)]].values.tolist()
        x_data = x_data[0]+x_data[1]
        y_data = rows.iloc[1][str(i+96)]
        x.append(x_data)
        y.append(y_data)

print(np.shape(np.array(x)))
print(np.shape(np.array(y)))


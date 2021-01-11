import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

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
    if cow=='All':
        continue
    for i in range(1,1657, 24):
        print(cow)
        print(i)
        rows = df.loc[(df["Cow"]==cow)]
        x_data = rows[[str(j) for j in range(i,i+96)]].values.tolist()
        x_data = x_data[1]+x_data[0]
        y_data = rows.iloc[1][str(i+102)]
        x.append(x_data)
        y.append(y_data)

x = np.array(x)
y = np.array(y)
print(np.shape(x))
print(np.shape(y))
print(x[0])
print(y[0])

X = sm.add_constant(x)
results = sm.OLS(y,X).fit()

print(results.summary())


coeffs = results.params
pvals = results.pvalues

fig, ax1 = plt.subplots()
ax1.plot(abs(coeffs))
ax2 = ax1.twinx()
ax2.plot(pvals, 'ro')

fig.tight_layout()
plt.show()

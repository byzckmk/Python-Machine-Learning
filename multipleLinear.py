
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# the first three are default for machine learning
from sklearn.linear_model import LinearRegression

data = pd.read_csv("insurance.csv")

print(data.columns)
# -> console -> data.describe()

# axis y
expenses = data.expenses.values.reshape(-1,1)

# axis x
ageBmis = data.iloc[:,[0,2]].values

regression = LinearRegression()
regression.fit(ageBmis,expenses)

print(regression.predict(np.array([[20,20],[20,21],[20,22],[20,23],[20,24]])))
print()
print(regression.predict(np.array([[30,20],[30,21],[20,22],[20,23],[20,24]])))


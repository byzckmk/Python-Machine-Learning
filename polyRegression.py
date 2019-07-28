
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# the first three are default for machine learning
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

data = pd.read_csv("positions.csv")

print(data.columns)
# -> console -> data.describe()

# axis y
salary = data.iloc[:,2].values.reshape(-1,1)

# axis x
level = data.iloc[:,1].values.reshape(-1,1)

regression = LinearRegression()
regression.fit(level,salary)

guess = regression.predict([[8.3]])
print(guess)

plt.scatter(level,salary,color="red")
plt.plot(level,regression.predict(level),color="blue")
plt.show()


regressionPoly = PolynomialFeatures(degree = 4)
levelPoly = regressionPoly.fit_transform(level)

regression2 = LinearRegression()
regression2.fit(levelPoly,salary)

guess2 = regression2.predict(regressionPoly.fit_transform([[8.3]]))
print()
print(guess2)

plt.scatter(level,salary,color="red")
plt.plot(level,regression2.predict(levelPoly),color="green")
plt.show()



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

data = pd.read_csv("positions.csv")

level = data.iloc[:,1].values.reshape(-1,1)
salary = data.iloc[:,2].values.reshape(-1,1)

regression = DecisionTreeRegressor()
regression.fit(level,salary)
print(regression.predict([[8.6]]))

plt.scatter(level,salary,color="red")
x = np.arange(min(level),max(level),0.01).reshape(-1,1)
plt.plot(x,regression.predict(x),color="orange")
plt.xlabel("level")
plt.ylabel("salary")
plt.title("Decision Tree Model")
plt.show()


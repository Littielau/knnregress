import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn import neighbors
from math import sqrt
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data"
abalone = pd.read_csv(url, header=None)
#print(abalone.head())
abalone.columns = ["Sex", "Length", "Diameter", "Height", "Whole weight", "Shucked weight", "Viscera weight", "Shell weight", "Rings"]
abalone = abalone.drop("Sex", axis=1)
X = abalone.drop("Rings", axis=1)
X = X.values
y = abalone["Rings"]
y = y.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12345)

model = neighbors.KNeighborsRegressor(n_neighbors=15)
model.fit(X_train, y_train)  #fit the model
pred=model.predict(X_test)

cmap = sns.cubehelix_palette(as_cmap=True)
f, ax = plt.subplots()
points = ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, s=50, cmap=cmap)
ax.set_title('Test data')
ax.set_xlabel('Length')
ax.set_ylabel('Diameter')
f.colorbar(points)
plt.show()

cmap = sns.cubehelix_palette(as_cmap=True)
f, ax = plt.subplots()
points = ax.scatter(X_test[:, 0], X_test[:, 1], c=pred, s=50, cmap=cmap)
ax.set_title('Prediction')
ax.set_xlabel('Length')
ax.set_ylabel('Diameter')
f.colorbar(points)
plt.show()

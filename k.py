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

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=12345)

rmse_val = [] #to store rmse values for different k
error1 = 5
for K in range(50):
    K = K+1
    model = neighbors.KNeighborsRegressor(n_neighbors = K)

    model.fit(X_train, y_train)  #fit the model
    pred=model.predict(X_test) #make prediction on test set
    error = sqrt(mean_squared_error(y_test,pred)) #calculate rmse
    rmse_val.append(error) #store rmse values
    #print('RMSE value for k= ' , K , 'is:', error)
    if error< error1:
        error1=error
        K1=K-1

# plotting the rmse values against k values
curve = pd.DataFrame(rmse_val)
curve.plot()
plt.xlabel('K')
plt.ylabel('rmse')
plt.show()
print(K1)

model = neighbors.KNeighborsRegressor(n_neighbors = 3)

model.fit(X_train, y_train)
pred=model.predict(X_train)
error = sqrt(mean_squared_error(y_train,pred))
print('K=3, test rmse', error)

pred=model.predict(X_test)
error = sqrt(mean_squared_error(y_test,pred))
print('K=3, train rmse',error)


model = neighbors.KNeighborsRegressor(n_neighbors = 15)

model.fit(X_train, y_train)
pred=model.predict(X_train)
error = sqrt(mean_squared_error(y_train,pred))
print('K=15, test rmse', error)

pred=model.predict(X_test)
error = sqrt(mean_squared_error(y_test,pred))
print('K=15, train rmse',error)
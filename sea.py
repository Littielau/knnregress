import pandas as pd
import numpy as np
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data"
abalone = pd.read_csv(url, header=None)
print(abalone.head())
abalone.columns = ["Sex", "Length", "Diameter", "Height", "Whole weight", "Shucked weight", "Viscera weight", "Shell weight", "Rings"]
abalone = abalone.drop("Sex", axis=1)
X = abalone.drop("Rings", axis=1)
X = X.values
y = abalone["Rings"]
y = y.values
print('Length 0.075-0.815')
Length = float(input())
print('Diameter 0.055-0.650')
Diameter = float(input())
print('Height 0.00-1.130')
Height = float(input())
print('Whole weight 0.002-2.826')
Whole_weight = float(input())
print('Shucked weight 0.001-1.488')
Shucked_weight = float(input())
print('Viscera weight 0.001-0.760')
Viscera_weight = float(input())
print('Shell weight 0.002-1.005')
Shell_weight = float(input())
new_data_point = [Length, Diameter, Height, Whole_weight, Shucked_weight, Viscera_weight,Shell_weight]
distances = np.linalg.norm(X - new_data_point, axis=1)
k = 15
nearest_neighbor_ids = distances.argsort()[:k]
print(nearest_neighbor_ids)
nearest_neighbor_rings = y[nearest_neighbor_ids]
print(nearest_neighbor_rings)
prediction = nearest_neighbor_rings.mean()
print(int(prediction))
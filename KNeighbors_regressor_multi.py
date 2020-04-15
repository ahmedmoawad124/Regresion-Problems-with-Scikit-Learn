'''********************************************************* 
   **********  KNN regression ( multi-features ) ***********
   ********************************************************* '''
import pandas as pd
import numpy as  np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

data = pd.read_csv("advertising.csv")

# showning the data
print("The first five rows:\n", data.head())
print("-" * 60)
data.info()
print("-" * 60)

# Preparing X and Y
X = data[['TV', 'Radio', 'Newspaper']].to_numpy()
Y = data['Sales'].to_numpy()

# spliting the data into training and testing data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size = 0.8, random_state=40)

print("The type of X train: ", type(X_train))
print("The type of Y train: ", type(Y_train))
print("-" * 60)

# choosing the K neighbors
error = []
m = 160
for i in range(m):
    regressor = KNeighborsRegressor(n_neighbors=i+1, weights= 'uniform', algorithm= 'kd_tree')
    regressor.fit(X_train, Y_train)
    acc = regressor.score(X_test, Y_test)
    single_error = 1- acc
    error.append(single_error)
    
k = range(1, m+1)
plt.plot(k, error)
plt.show()

# K-Neighbors Regressor model
KNN_regressor = KNeighborsRegressor(n_neighbors= 3, weights= 'uniform', algorithm= 'kd_tree')
KNN_regressor.fit(X_train, Y_train)


# The prediction:
y_predicted = KNN_regressor.predict(X_test)
y_from_hypothis = KNN_regressor.predict(X_train)


# Accuracy
acc1 = KNN_regressor.score(X_train, Y_train)
print("The accuracy of traing set = ", acc1)
acc2 = KNN_regressor.score(X_test, Y_test)
print("The accuracy of testing set = ", acc2)


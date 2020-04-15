'''********************************************************* 
   *********  SVM regression ( single-feature ) ************
   ********************************************************* '''

import pandas as pd
import numpy as  np
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import operator

data = pd.read_csv("tvmarketing.csv")

# showning the data
print("The first five rows:\n", data.head())
print("-" * 60)
data.info()
print("-" * 60)

# Preparing X and Y
X = data['TV'].to_numpy()
Y = data['Sales'].to_numpy()

# spliting the data into training and testing data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size = 0.8, random_state=40)

print("The type of X train: ", type(X_train))
print("The type of Y train: ", type(Y_train))
print("-" * 60)

X_train = X_train.reshape(-1, 1)
X_test = X_test.reshape(-1, 1)



# SV-Regressor model
svm_regressor = SVR(kernel = 'rbf', C= 1.5)  # Radial-basis function kernel
                                             #It is also known as the “squared exponential” kernel
svm_regressor.fit(X_train, Y_train)


# The prediction:
y_predicted = svm_regressor.predict(X_test)
y_from_hypothis = svm_regressor.predict(X_train)


# Accuracy
acc1 = svm_regressor.score(X_train, Y_train)
print("The accuracy of traing set = ", acc1)
acc2 = svm_regressor.score(X_test, Y_test)
print("The accuracy of testing set = ", acc2)

# Vesualization training set:
plt.scatter(X_train, Y_train, color='red')

# sort the values of X_train before line plot
sort_axis = operator.itemgetter(0)
sorted_zip = sorted(zip(X_train,y_from_hypothis), key=sort_axis)
X_train, y_from_hypothis = zip(*sorted_zip)


plt.plot(X_train, y_from_hypothis, color='blue')
plt.show()

# Vesualization testing set:
plt.scatter(X_test, Y_test, color='red')

# sort the values of X_train before line plot
sort_axis = operator.itemgetter(0)
sorted_zip = sorted(zip(X_test,y_predicted), key=sort_axis)
X_test, y_predicted = zip(*sorted_zip)


plt.plot(X_test, y_predicted, color='blue')
plt.show()
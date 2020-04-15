'''********************************************************* 
   ***************  polynomial regression ******************
   ********************************************************* '''
import numpy as  np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import operator

np.random.seed(0)
X = 2 - 3 * np.random.normal(0, 1, 100)
Y = X - 2 * (X ** 2) + 0.5 * (X ** 3) + np.random.normal(-3, 3, 100)
plt.scatter(X,Y, s=10)
plt.show()

# spliting the data into training and testing data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size = 0.8)

X_train = X_train.reshape(-1, 1)
X_test = X_test.reshape(-1, 1)

print("The type of X train: ", type(X_train))
print("The type of Y train: ", type(Y_train))

polynomial_features = PolynomialFeatures(degree = 3)
X_train_poly = polynomial_features.fit_transform(X_train)
X_test_poly = polynomial_features.fit_transform(X_test)

# SkLearn regression model
regressor = LinearRegression()
regressor.fit(X_train_poly, Y_train)

# print the coefficients
print("The intercept = ",regressor.intercept_)
print("The coef = ",regressor.coef_)

# The prediction:
y_predicted = regressor.predict(X_test_poly)
y_from_hypothis = regressor.predict(X_train_poly)

# Vesualization:
plt.scatter(X_train, Y_train, color='red')

# sort the values of X_train before line plot
sort_axis = operator.itemgetter(0)
sorted_zip = sorted(zip(X_train,y_from_hypothis), key=sort_axis)
X_train, y_from_hypothis = zip(*sorted_zip)

plt.plot(X_train, y_from_hypothis, color='blue')
plt.show()

# Accuracy
acc1 = regressor.score(X_train_poly, Y_train)
print("The accuracy of traing set = ", acc1)
acc2 = regressor.score(X_test_poly, Y_test)
print("The accuracy of testing set = ", acc2)
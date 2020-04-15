'''********************************************************* 
   *********  Linear regression ( multi-features ) *********
   ********************************************************* '''
import pandas as pd
import numpy as  np
from sklearn.linear_model import LinearRegression
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

# SkLearn regression model
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# print the coefficients
print("The intercept = ",regressor.intercept_)
print("The coef = ",regressor.coef_)

# Accuracy
acc1 = regressor.score(X_train, Y_train)
print("The accuracy of traing set = ", acc1)
acc2 = regressor.score(X_test, Y_test)
print("The accuracy of testing set = ", acc2)
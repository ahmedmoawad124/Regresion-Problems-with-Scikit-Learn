'''********************************************************* 
   **********  NN regression ( multi-features ) ***********
   ********************************************************* '''
import pandas as pd
import numpy as  np
from sklearn.neural_network import MLPRegressor
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

# NN Regressor model
NN_regressor = MLPRegressor(hidden_layer_sizes=(5,), activation='identity', solver='lbfgs', learning_rate='constant', max_iter=100, random_state= 40)
                        # hidden_layer_sizes=(hidden units, hidden layer)
NN_regressor.fit(X_train, Y_train)


# The prediction:
y_predicted = NN_regressor.predict(X_test)
y_from_hypothis = NN_regressor.predict(X_train)


# Accuracy
acc1 = NN_regressor.score(X_train, Y_train)
print("The accuracy of traing set = ", acc1)
acc2 = NN_regressor.score(X_test, Y_test)
print("The accuracy of testing set = ", acc2)
import numpy as np
import pandas as pd

# Load the dataset
from sklearn.datasets import load_linnerud

linnerud_data = load_linnerud()
X = linnerud_data.data
y = linnerud_data.target

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error as mse
from sklearn.linear_model import LinearRegression

# TODO: split the data into training and testing sets,
# using the standard settings for train_test_split.
# Then, train and test the classifiers with your newly split data instead of X and y.
from sklearn import cross_validation
X, x_test, y, y_test = cross_validation.train_test_split(X, y)

reg1 = DecisionTreeRegressor()
reg1.fit(X, y)
mse_dt = mse(y_test, reg1.predict(x_test))
print "Decision Tree mean squared error: {:.2f}".format(mse_dt)

reg2 = LinearRegression()
reg2.fit(X, y)
mse_ln = mse(y, reg2.predict(X))
print "Linear regression mean squared error: {:.2f}".format(mse_ln)

results = {
 "Linear Regression": mse_ln,
 "Decision Tree": mse_dt
}
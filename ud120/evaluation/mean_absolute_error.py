import numpy as np
import pandas as pd

# Load the dataset
from sklearn.datasets import load_linnerud

linnerud_data = load_linnerud()
X = linnerud_data.data
y = linnerud_data.target

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error as mae
from sklearn.linear_model import LinearRegression

# TODO: split the data into training and testing sets,
# using the standard settings for train_test_split.
# Then, train and test the classifiers with your newly split data instead of X and y.
from sklearn import cross_validation
X, x_test, y, y_test = cross_validation.train_test_split(X, y)

reg = DecisionTreeRegressor()
reg.fit(X, y)
mae_dt = mae(reg.predict(x_test),y_test)
print "Decision Tree mean absolute error: {:.2f}".format(mae_dt)

reg = LinearRegression()
reg.fit(X, y)
mae_ln = mae(reg.predict(X),y)
print "Linear regression mean absolute error: {:.2f}".format(mae_ln)

results = {
 "Linear Regression": mae_ln,
 "Decision Tree": mae_dt
}
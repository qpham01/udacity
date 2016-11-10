# Import libraries necessary for this project
import numpy as np
import pandas as pd
from IPython.display import display # Allows the use of display() for DataFrames


# Load the wholesale customers dataset
try:
    data = pd.read_csv("customers.csv")
    data.drop(['Region', 'Channel'], axis = 1, inplace = True)
    print "Wholesale customers dataset has {} samples with {} features each.".format(*data.shape)
except:
    print "Dataset could not be loaded. Is the dataset missing?"

# Note:  I decided to score the relevance of all features, not just one.
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeRegressor
random_n=999
regressor = DecisionTreeRegressor(random_state=random_n)

def show_samples(df, label, target):
    indices = [0, 1, 2]
    samples = pd.DataFrame(df.iloc[indices], columns = df.keys()).reset_index(drop = True)
    print "Samples of", label, "dataset for target column", target, ":"
    display(samples)    
    
# TODO: Make a copy of the DataFrame, using the 'drop' function to drop the given feature
columns = [ 'Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicatessen' ]
for column in columns:
    new_data = data.drop(column, axis = 1)
    
    target = data[[column]]
    
    show_samples(target, "Target", column)
        
    # TODO: Split the data into training and testing sets using the given feature as the target
    X_train, X_test, y_train, y_test = train_test_split(new_data, target, test_size=0.25, random_state=random_n)

    # TODO: Create a decision tree regressor and fit it to the training set
    regressor.fit(X_train, y_train)

    # TODO: Report the score of the prediction using the testing set
    score = regressor.score(X_test, y_test)    

    print "Score for", column, ":", score    
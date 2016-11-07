'''
Student Intervention Exercise
'''
# Import libraries
import numpy as np
import pandas as pd
from time import time
from sklearn.metrics import f1_score

# Read student data
student_data = pd.DataFrame(pd.read_csv("student-data.csv"))
print "Student data read successfully!"

# TODO: Calculate number of students
n_students = len(student_data.index)

# TODO: Calculate number of features
n_features = len(student_data.columns)

# TODO: Calculate passing students
PASSED = student_data[student_data['passed'] == 'yes']
n_passed = float(len(PASSED))

# TODO: Calculate failing students
FAILED = student_data[student_data['passed'] == 'no']
n_failed = float(len(FAILED))

# TODO: Calculate graduation rate
grad_rate = 100.0 * n_passed / n_students

# Print the results
print "Total number of students: {}".format(n_students)
print "Number of features: {}".format(n_features)
print "Number of students who passed: {}".format(n_passed)
print "Number of students who failed: {}".format(n_failed)
print "Graduation rate of the class: {:.2f}%".format(grad_rate)

# Extract feature columns
feature_cols = list(student_data.columns[:-1])

# Extract target column 'passed'
target_col = student_data.columns[-1] 

# Show the list of columns
print "Feature columns:\n{}".format(feature_cols)
print "\nTarget column: {}".format(target_col)

# Separate the data into feature data and target data (X_all and y_all, respectively)
X_all = student_data[feature_cols]
y_all = student_data[target_col]

# Show the feature information by printing the first five rows
print "\nFeature values:"
print X_all.head()

def preprocess_features(X):
    ''' Preprocesses the student data and converts non-numeric binary variables into
        binary (0/1) variables. Converts categorical variables into dummy variables. '''
    
    # Initialize new output DataFrame
    output = pd.DataFrame(index = X.index)

    # Investigate each feature column for the data
    for col, col_data in X.iteritems():
        
        # If data type is non-numeric, replace all yes/no values with 1/0
        if col_data.dtype == object:
            col_data = col_data.replace(['yes', 'no'], [1, 0])

        # If data type is categorical, convert to dummy variables
        if col_data.dtype == object:
            # Example: 'school' => 'school_GP' and 'school_MS'
            col_data = pd.get_dummies(col_data, prefix = col)  
        
        # Collect the revised columns
        output = output.join(col_data)
    
    return output

X_all = preprocess_features(X_all)
print "Processed feature columns ({} total features):\n{}".format(len(X_all.columns), list(X_all.columns))

'''
Set up training set and test set
'''
# TODO: Import any additional functionality you may need here
from sklearn.cross_validation import train_test_split

# TODO: Set the number of training points
num_train = 300

# Set the number of testing points
num_test = X_all.shape[0] - num_train

# TODO: Shuffle and split the data into training and testing subsets
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=num_test, random_state=222)

# Show the results of the split
print "Training set has {} samples.".format(X_train.shape[0])
print "Testing set has {} samples.".format(X_test.shape[0])
def train_classifier(clf, X_train, y_train):
    ''' Fits a classifier to the training data. '''
    
    # Start the clock, train the classifier, then stop the clock
    start = time()
    clf.fit(X_train, y_train)
    end = time()
    
    # Print the results
    print "Trained model in {:.4f} seconds".format(end - start)

    
def predict_labels(clf, features, target):
    ''' Makes predictions using a fit classifier based on F1 score. '''
    
    # Start the clock, make predictions, then stop the clock
    start = time()
    y_pred = clf.predict(features)
    end = time()
    
    # Print and return results
    print "Made predictions in {:.4f} seconds.".format(end - start)
    return f1_score(target.values, y_pred, pos_label='yes')


def train_predict(clf, X_train, y_train, X_test, y_test):
    ''' Train and predict using a classifer based on F1 score. '''
    
    # Indicate the classifier and the training set size
    print "Training a {} using a training set size of {}. . .".format(clf.__class__.__name__, len(X_train))
    
    # Train the classifier
    train_classifier(clf, X_train, y_train)
    
    # Print the results of prediction for both training and testing
    print "F1 score for training set: {:.4f}.".format(predict_labels(clf, X_train, y_train))
    print "F1 score for test set: {:.4f}.".format(predict_labels(clf, X_test, y_test))

# TODO: Import the three supervised learning models from sklearn
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegressionCV

# TODO: Initialize the three models
clf_A = SVC(random_state=1001)
#clf_B = AdaBoostClassifier(n_estimators=30, random_state=1001)
#clf_B = GradientBoostingClassifier(random_state=1001)
#clf_A = KNeighborsClassifier()
#clf_B = DecisionTreeClassifier(random_state=1001) 
#clf_C = RandomForestClassifier(random_state=1001)
#clf_C = GaussianNB()
#clf_C = SGDClassifier(random_state=1001)
clf_B = LogisticRegressionCV(random_state=1001)
#clf_C = AdaBoostClassifier(random_state=1001)
clf_C = BaggingClassifier(random_state=1001)

# TODO: Set up the training set sizes
X_train_100 = X_train[0:100]
y_train_100 = y_train[0:100]

X_train_200 = X_train[0:200]
y_train_200 = y_train[0:200]

X_train_300 = X_train[0:300]
y_train_300 = y_train[0:300]

# TODO: Execute the 'train_predict' function for each classifier and each training set size
train_predict(clf_A, X_train_100, y_train_100, X_test, y_test)
train_predict(clf_A, X_train_200, y_train_200, X_test, y_test)
train_predict(clf_A, X_train_300, y_train_300, X_test, y_test)

# TODO: Execute the 'train_predict' function for each classifier and each training set size
train_predict(clf_B, X_train_100, y_train_100, X_test, y_test)
train_predict(clf_B, X_train_200, y_train_200, X_test, y_test)
train_predict(clf_B, X_train_300, y_train_300, X_test, y_test)

# TODO: Execute the 'train_predict' function for each classifier and each training set size
train_predict(clf_C, X_train_100, y_train_100, X_test, y_test)
train_predict(clf_C, X_train_200, y_train_200, X_test, y_test)
train_predict(clf_C, X_train_300, y_train_300, X_test, y_test)

print clf_C

# TODO: Import 'GridSearchCV' and 'make_scorer'
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import make_scorer, f1_score


'''
#AdaBoostClassifier
# TODO: Create the parameters list you wish to tune
parameters = { 'base_estimator' : [LogisticRegressionCV(), DecisionTreeClassifier()],
               'learning_rate' : [0.001, 0.01, 0.1, 1.0],
               'n_estimators' : [25, 50, 75, 100] }

# TODO: Initialize the classifier
# base_clf = LogisticRegressionCV()
main_clf = AdaBoostClassifier()
'''
'''
#GradientBoosting
# TODO: Create the parameters list you wish to tune
parameters = { 'max_depth' : [1, 2, 3, 4], 
               'learning_rate' : [0.008, 0.01, 0.02, 0.04],
               'n_estimators' : [50, 100, 150, 200] }
               #'min_samples_split' : [1, 2, 3, 4, 5],
               #'min_samples_leaf' : [1, 2, 3] }
# TODO: Initialize the classifier
main_clf = GradientBoostingClassifier(random_state=1001)
'''
#'''
#Bagging
#'
parameters = { 'n_estimators' : [ 5, 10, 15],
               'max_samples' : [0.1, 0.2, 0.3],
               'max_features' : [0.6, 0.8, 1.0] }

# TODO: Initialize the classifier
base_clf = LogisticRegressionCV(random_state=1001, solver='liblinear')
main_clf = BaggingClassifier(base_estimator=base_clf, random_state=1001)
#'''
'''
#LogisticRegressionCV
parameters = { 'solver' : ['newton-cg', 'lbfgs', 'liblinear', 'sag'] }

# TODO: Initialize the classifier
main_clf = LogisticRegressionCV(random_state=1001, penalty='l2', cv=3)
'''
'''
#SVM
parameters = { 'kernel' : ['linear', 'poly', 'rbf', 'sigmoid'],
              'tol' : [0.001, 0.00001, .0000001] }

# TODO: Initialize the classifier
main_clf = SVC(random_state=1001)
'''

# TODO: Make an f1 scoring function using 'make_scorer' 
f1_scorer = make_scorer(f1_score, pos_label='yes')

# TODO: Perform grid search on the classifier using the f1_scorer as the scoring method
grid_obj = GridSearchCV(estimator=main_clf, param_grid=parameters, scoring=f1_scorer, verbose=3)

# TODO: Fit the grid search object to the training data and find the optimal parameters
t0 = time()
grid_obj = grid_obj.fit(X_train, y_train)
print ''
print 'Grid search took {:.2f} seconds.'.format(time() - t0)

# Get the estimator
clf = grid_obj.best_estimator_

# Report the final F1 score for training and testing after parameter tuning
print "Tuned model has a training F1 score of {:.4f}.".format(predict_labels(clf, X_train, y_train))
print "Tuned model has a testing F1 score of {:.4f}.".format(predict_labels(clf, X_test, y_test))
print ''
print clf
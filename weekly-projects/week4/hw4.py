import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# load data
raw_data = pd.read_csv('compas-scores-two-years.csv')
# print a list of variable names
print(raw_data.columns)
# look at the first 5 rows
raw_data.head(5)

# Select features and response variables

# Features by type
numerical_features = ['juv_misd_count', 'juv_other_count', 'juv_fel_count', 'priors_count', 'age']
binary_categorical_features = ['sex', 'c_charge_degree']
other_categorical_features = ['race']
all_features = binary_categorical_features + other_categorical_features + numerical_features

# Possible esponse variables
response_variables = ['is_recid', 'is_violent_recid', 'two_year_recid']

# Variables that are used for data cleaning
check_variables = ['days_b_screening_arrest']

# Subselect data
df = raw_data[all_features+response_variables+check_variables]

# Apply filters
df = df[(df['days_b_screening_arrest'] <= 30) &
        (df['days_b_screening_arrest'] >= -30) &
        (df['is_recid'] != -1) &
        (df['c_charge_degree'] != 'O')]

df = df[all_features+response_variables]
print('Dataframe has {} rows and {} columns.'.format(df.shape[0], df.shape[1]))

# Code binary features as 0 and 1
for x in binary_categorical_features:
    for new_value, value in enumerate(set(df[x])):
        print("Replace {} with {}.".format(value, new_value))
        df = df.replace(value, new_value)


# Use 1-hot encoding for other categorical variables
one_hot_features = []
for x in other_categorical_features:
    for new_feature, value in enumerate(set(df[x])):
        feature_name = "{}_is_{}".format(x,value)
        df.insert(3, feature_name, df[x]==value)
        one_hot_features += [feature_name]

# Check what the data frame looks like now
df.head(10)

# list of features
features = numerical_features + binary_categorical_features + one_hot_features

# features data frame
X = df[features]

# responses data frame
Y = df[response_variables]

# HW4 Part 1 Step 4
# Split the data into a training set containing 90% of the data
# and test set containing 10% of the data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)

# HW4 Part 2 Step 1
# Create a model
dtc = DecisionTreeClassifier()

# Fit model to training data
dtc.fit(X_train, Y_train['two_year_recid'])

# Evaluate training accuracy
accuracy = dtc.score(X_train, Y_train['two_year_recid'])

# Check size of decision tree
num_leaves = dtc.get_n_leaves()

# Report results
print('Trained decision tree with {} leaves and training accuracy {:.2f}.'.format(num_leaves, accuracy))

# HW4 Part 2 Step 2

# Perform 5-fold cross-validation for different tree sizes
print('Leaves\tMean accuracy')
print('---------------------')
for num_leaves in range(100, 1800, 100):

    # Trees must have at least 2 leaves
    if num_leaves >= 2:
        # Construct a classifier with a limit on its number of leaves
        dtc = DecisionTreeClassifier(max_leaf_nodes=num_leaves, random_state=42)

        # Get validation accuracy via 5-fold cross-validation
        scores = cross_val_score(dtc, X, Y['two_year_recid'], cv=5)

    print("{}\t{:.3f}".format(num_leaves, scores.mean()))

# Perform for a different sizes to test values for maximum leaf nodes
print('Leaves\tMean accuracy')
print('---------------------')

# Adjusted range for max_leaf_nodes
best_mean_accuracy = 0
best_num_leaves = 0

for num_leaves in range(50, 2000, 50):

    # Trees must have at least 2 leaves
    if num_leaves >= 2:

        # Construct a classifier with a limit on its number of leaves
        dtc = DecisionTreeClassifier(max_leaf_nodes=num_leaves, random_state=42)

        # Get validation accuracy via 5-fold cross-validation
        scores = cross_val_score(dtc, X, Y['two_year_recid'], cv=5)

        mean_score = scores.mean()

        if mean_score > best_mean_accuracy:
            best_mean_accuracy = mean_score
            best_num_leaves = num_leaves

    print("{}\t{:.3f}".format(num_leaves, mean_score))

print("\nBest number of leaves: {}\tMean accuracy: {:.3f}".format(best_num_leaves, best_mean_accuracy))


# HW4 Part 2 Step 3
# Selected value of max_leaf_nodes based on the previous cross-validation results
selected_max_leaf_nodes = best_num_leaves

# Create a model
dtc = DecisionTreeClassifier(max_leaf_nodes=selected_max_leaf_nodes, random_state=42)

# Fit model to training data
dtc.fit(X_train, Y_train['two_year_recid'])

# Evaluate test accuracy
accuracy = dtc.score(X_test, Y_test['two_year_recid'])

# Check size of decision tree
num_leaves = dtc.get_n_leaves()

# Report results
print('Trained decision tree with {} leaves and test accuracy {:.2f}.'.format(num_leaves, accuracy))

# HW4 Part 3 Step 1
# Create subset of training data without information on race.
# (The information on race was encoded in the one-hot features.)
remaining_features = [v for v in X.columns if v not in one_hot_features]
X_train_sub = X_train[remaining_features]
X_test_sub = X_test[remaining_features]

# Create a model
dtc = DecisionTreeClassifier(max_leaf_nodes=39)

# Fit model to training data
dtc.fit(X_train_sub, Y_train['two_year_recid'])

# Evaluate training accuracy
y_pred = dtc.predict(X_test_sub)
accuracy = (y_pred == Y_test['two_year_recid']).mean()

# Check size of decision tree
num_leaves = dtc.get_n_leaves()

# Report results
print('Trained decision tree with {} leaves and test accuracy {:.2f}.'.format(num_leaves, accuracy))

# Comparing the mean accuracy values of the set with all features to this subset, I notice that racial factors do
# influence the outcome, however the difference is minimal. The test accuracy for the full set was .64 and the test
# accuracy for the subset was .65, indicating that although the removal of racial features improves the accuracy
# of the test, it does so minimally. Thus, there is the possibility of racial bias included, however it is small due to
# the small difference in accuracies.


# HW 4 Part 3 Step 2
# Create subset of training data without information on race
remaining_features = [v for v in X.columns if v not in one_hot_features]
X_train_sub = X_train[remaining_features]
X_test_sub = X_test[remaining_features]

# Create a model with all features
dtc_full = DecisionTreeClassifier(max_leaf_nodes=39, random_state=42)
dtc_full.fit(X_train, Y_train['two_year_recid'])

# Create a model without race features
dtc_sub = DecisionTreeClassifier(max_leaf_nodes=39, random_state=42)
dtc_sub.fit(X_train_sub, Y_train['two_year_recid'])

# Export the decision rules to text
tree_rules_full = export_text(dtc_full, feature_names=list(X.columns))
tree_rules_sub = export_text(dtc_sub, feature_names=remaining_features)

print("Decision rules for the tree with all features:\n")
print(tree_rules_full)

print("\nDecision rules for the tree without racial information:\n")
print(tree_rules_sub)

# The first decision tree indicates the possibility for racial bias due to the splits that occurs based on race,
# indicating that there is the ability for racial bias if using biased datasets. However, due to the analysis performed
# in step 1, there is little indication of racial bias.

# HW 4 Part 4 Step 1
# Fit an LDA and Logistic Regression

# Create and fit the LDA model
lda = LinearDiscriminantAnalysis()
lda.fit(X_train_sub, Y_train['two_year_recid'])

# Predict on the test set
lda_predictions = lda.predict(X_test_sub)

# Evaluate the LDA model
lda_accuracy = accuracy_score(Y_test['two_year_recid'], lda_predictions)
print('LDA Test Accuracy: {:.2f}'.format(lda_accuracy))


# Create and fit the Logistic Regression model
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_sub, Y_train['two_year_recid'])

# Predict on the test set
log_reg_predictions = log_reg.predict(X_test_sub)

# Evaluate the Logistic Regression model
log_reg_accuracy = accuracy_score(Y_test['two_year_recid'], log_reg_predictions)
print('Logistic Regression Test Accuracy: {:.2f}'.format(log_reg_accuracy))


# HW 4 Part 4 Step 2
# Tune and fit ensemble methods

# Random Forest model
# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Create Random Forest model
rf = RandomForestClassifier()

# Perform Grid Search
grid_search_rf = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search_rf.fit(X_train_sub, Y_train['two_year_recid'])

# Get best parameters and fit model
best_rf = grid_search_rf.best_estimator_

# Predict and evaluate
rf_predictions = best_rf.predict(X_test_sub)
rf_accuracy = accuracy_score(Y_test['two_year_recid'], rf_predictions)
print('Best Random Forest Test Accuracy: {:.2f}'.format(rf_accuracy))
print('Best Parameters:', grid_search_rf.best_params_)

# Gradient Boosting model
# Define parameter grid
param_grid_gb = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10]
}

# Create Gradient Boosting model
gb = GradientBoostingClassifier()

# Perform Grid Search
grid_search_gb = GridSearchCV(estimator=gb, param_grid=param_grid_gb, cv=5, n_jobs=-1, verbose=2)
grid_search_gb.fit(X_train_sub, Y_train['two_year_recid'])

# Get best parameters and fit model
best_gb = grid_search_gb.best_estimator_

# Predict and evaluate
gb_predictions = best_gb.predict(X_test_sub)
gb_accuracy = accuracy_score(Y_test['two_year_recid'], gb_predictions)
print('Best Gradient Boosting Test Accuracy: {:.2f}'.format(gb_accuracy))
print('Best Parameters:', grid_search_gb.best_params_)


# HW 4 Part 4 Step 3
# Tune and fit SVC

# Define a small parameter grid for GridSearchCV because it would not run on my machine
param_grid = {
    'C': [0.1, 1, 10],
    'gamma': [0.1, 0.01],
    'kernel': ['linear', 'rbf']
}

# Create a GridSearchCV object with SVC and the reduced parameter grid
svc_grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=2, cv=3)

# Fit the model to the training data
svc_grid.fit(X_train_sub, Y_train['two_year_recid'])

# Print the best parameters found by GridSearchCV
print("Best parameters found: ", svc_grid.best_params_)

# Predict on the test set using the best found model
svc_predictions = svc_grid.best_estimator_.predict(X_test_sub)

# Evaluate the SVC model
svc_accuracy = accuracy_score(Y_test['two_year_recid'], svc_predictions)
print('SVC Test Accuracy: {:.2f}'.format(svc_accuracy))

# HW 4 Part 4 Step 4


# Compare test accuracy of all your models
def compare_model_accuracies(accuracies):
    best_model = max(accuracies, key=accuracies.get)
    best_accuracy = accuracies[best_model]
    print('Model Test Accuracies:')
    for model, accuracy in accuracies.items():
        print(f'{model}: {accuracy:.2f}')
    print(f'\nBest Model: {best_model}\tTest Accuracy: {best_accuracy:.2f}')


# Accuracies
accuracies = {
    'Decision Tree': 0.65,
    'LDA': 0.68,
    'Logistic Regression': 0.67,
    'SVC': 0.67,
    'Random Forest': 0.67,
    'Gradient Boosting': 0.66
}

# Compare the accuracies
compare_model_accuracies(accuracies)

# The LDA is the model with the best test accuracy

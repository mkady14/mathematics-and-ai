import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.linear_model import LinearRegression

# Read-in the diabetes dataset as a pandas DataFrame
diabetes = datasets.load_diabetes(as_frame=True)

# Get independent variables
X = diabetes.data

# Get dependent variable
y = diabetes.target

# Let's look at the data
X.describe()

# initialize model
model = LinearRegression()

# get variable names from column header in the data frame
var_names = X.columns

# select first variable
var_name1 = var_names[0]

# Modified by Madelyn Kady!
# select data associated with the first variable
x1 = X[var_name1]

# turn that dataframe column into a nx1 numpy array
x1_data = np.array([x1.to_numpy()]).T

# fit model
_ = model.fit(x1_data, y.to_numpy())

# get model predictions for each x value
yHat = model.predict(x1_data)

# get residuals
resid = yHat-y

# get R2 value
R2 = model.score(x1_data, y)
print('R2', R2)

# make a plot
plt.subplot(111)

# plot data
plt.plot(x1, y, marker='x', lw=0, color='blue')

# plot fit
plt.plot(x1, yHat, ls='--', color='red')

plt.xlabel(var_name1)
plt.ylabel('Target')
plt.show()

# HW 2 Part A step 2

# Dictionary to store the results
results = {}

# Loop through each independent variable
for var_name in X.columns:
    # Select data associated with the variable
    x = X[var_name]

    # Turn that dataframe column into a nx1 numpy array
    x_data = np.array([x.to_numpy()]).T

    # Fit model
    model.fit(x_data, y.to_numpy())

    # Get model predictions for each x value
    yHat = model.predict(x_data)

    # Calculate residuals
    resid = yHat - y.to_numpy()

    # Calculate statistics
    n = len(y)
    p = 1  # Number of predictors

    RSS = np.sum(resid ** 2)
    MSE = RSS / n
    RSE = np.sqrt(RSS / (n - p - 1))
    R2 = model.score(x_data, y.to_numpy())

    # Calculate t-statistic for the estimated model parameters
    se = np.sqrt(np.sum((resid) ** 2) / (n - 2)) / np.sqrt(np.sum((x - np.mean(x)) ** 2))
    t_stat = model.coef_[0] / se

    # Store the results
    results[var_name] = {
        't_stat': t_stat,
        'RSS': RSS,
        'MSE': MSE,
        'RSE': RSE,
        'R2': R2,
        'x_data': x,
        'yHat': yHat
    }

# Find the variable with the best R2
best_var = max(results, key=lambda k: results[k]['R2'])
best_result = results[best_var]

# Print the results for the best model
print("Best Variable: ", best_var)
print("t-statistic: ", best_result['t_stat'])
print("RSS: ", best_result['RSS'])
print("MSE: ", best_result['MSE'])
print("RSE: ", best_result['RSE'])
print("R2: ", best_result['R2'])


# Plot data
plt.plot(best_result['x_data'], y, marker='x', lw=0, color='blue')

# Plot fit
plt.plot(best_result['x_data'], best_result['yHat'], ls='--', color='red')

# Show the plot
plt.xlabel(best_var)
plt.ylabel('Target')
plt.title(f'Linear Regression Fit: {best_var} vs Target\nR2 = {best_result["R2"]:.2f}')
plt.show()

# HW 2 Part A step 3

# Get model fit
model.fit(X, y)

# Get model predictions
yHat = model.predict(X)

# Calculate residuals
resid = y - yHat

# Calculate statistics
n = len(y)
p = X.shape[1]  # Number of predictors

RSS = np.sum(resid**2)
TSS = np.sum((y - np.mean(y))**2)
MSE = RSS / n
RSE = np.sqrt(RSS / (n - p - 1))
R2 = model.score(X, y)

# Calculate F-statistic
MSR = (TSS - RSS) / p
MSE_reg = RSS / (n - p - 1)
F_stat = MSR / MSE_reg

# Print the results
print("F-statistic: ", F_stat)
print("RSS: ", RSS)
print("MSE: ", MSE)
print("RSE: ", RSE)
print("R2: ", R2)

# HW 2 Part A Step 4

# Fit the full multivariate linear model
full_model = LinearRegression()
full_model.fit(X, y)

# Calculate statistics for the full model
yHat_full = full_model.predict(X)
RSS_full = np.sum((y - yHat_full) ** 2)
TSS = np.sum((y - np.mean(y)) ** 2)
n = len(y)
p = X.shape[1]  # Number of predictors

# Dictionary to store F-statistics for each variable
f_statistics = {}

# Loop through each independent variable
for var_name in X.columns:
    # Create the reduced model by dropping the current variable
    X_reduced = X.drop(columns=[var_name])

    # Fit the reduced model
    reduced_model = LinearRegression()
    reduced_model.fit(X_reduced, y)

    # Calculate statistics for the reduced model
    yHat_reduced = reduced_model.predict(X_reduced)
    RSS_reduced = np.sum((y - yHat_reduced) ** 2)

    # Calculate the F-statistic
    f_stat = ((RSS_reduced - RSS_full) / (RSS_full / (n - p - 1))) * (n - p - 1)
    f_statistics[var_name] = f_stat

# Sort variables by their F-statistics in descending order
sorted_f_statistics = sorted(f_statistics.items(), key=lambda item: item[1], reverse=True)

# Display the top 3 variables with the highest F-statistics
top_3_variables = sorted_f_statistics[:3]
print("Top 3 variables by F-statistics:")
for var, f_stat in top_3_variables:
    print("Variable: ", var, "F-statistic: ", f_stat)

# Compare with the single-variable model rankings
single_variable_r2 = {var_name: results[var_name]['R2'] for var_name in X.columns}
sorted_single_variable_r2 = sorted(single_variable_r2.items(), key=lambda item: item[1], reverse=True)

print("\nTop 3 variables by single-variable R2:")
for var, r2 in sorted_single_variable_r2[:3]:
    print("Variable:", var, "R2:", r2)

# Analysis of differences in rankings
print("\nAnalysis of differences in rankings:")
for i in range(3):
    var_f = top_3_variables[i][0]
    var_r2 = sorted_single_variable_r2[i][0]
    if var_f != var_r2:
        print("Difference at rank", (i + 1), "F-statistic top variable is", var_f,
              "but single-variable R2 top variable is", var_r2)
    else:
        print("Same at rank", (i + 1), var_f)

# The three vars with the best F value are the three vars with the best-fitting single-variable
# however not in the same order, due to potential problems with multicollinearity, interaction terms
# and model complexity

# HW 2 Part 1 Step 5

from sklearn.model_selection import train_test_split

X = diabetes.data
y = diabetes.target

# Split the dataset into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the full multivariate linear model on the training set
full_model = LinearRegression()
full_model.fit(X_train, y_train)


# Function to calculate statistics
def calculate_statistics(model, X, y, n, p):
    y_hat_tts = model.predict(X)
    RSS_tts = np.sum((y - y_hat_tts) ** 2)
    R2_tts = model.score(X, y)
    RSE_tts = np.sqrt(RSS / (n - p - 1))
    MSE_tts = RSS_tts / n
    return RSS_tts, R2_tts, RSE_tts, MSE_tts


# Initialize storage for results
train_results = {}
test_results = {}

# Single-variable models
for var_name in X_train.columns:
    # Create the training data for the single-variable model
    x_train = X_train[[var_name]]
    x_test = X_test[[var_name]]

    # Fit the single-variable model on the training data
    model = LinearRegression()
    model.fit(x_train, y_train)

    # Calculate statistics for the training set
    RSS_train, R2_train, RSE_train, MSE_train = calculate_statistics(model, x_train, y_train, len(y_train), 1)
    train_results[var_name] = {
        'RSS': RSS_train,
        'R2': R2_train,
        'RSE': RSE_train,
        'MSE': MSE_train
    }

    # Calculate statistics for the test set
    RSS_test, R2_test, RSE_test, MSE_test = calculate_statistics(model, x_test, y_test, len(y_test), 1)
    test_results[var_name] = {
        'RSS': RSS_test,
        'R2': R2_test,
        'RSE': RSE_test,
        'MSE': MSE_test
    }

# Full multivariate model
full_model = LinearRegression()
full_model.fit(X_train, y_train)

# Calculate statistics for the full multivariate model on the training set
RSS_train, R2_train, RSE_train, MSE_train = calculate_statistics(full_model, X_train, y_train, len(y_train),
                                                                 X_train.shape[1])

# Calculate statistics for the full multivariate model on the test set
RSS_test, R2_test, RSE_test, MSE_test = calculate_statistics(full_model, X_test, y_test, len(y_test), X_test.shape[1])

# Store results for the full model
train_results['Full Model'] = {
    'RSS': RSS_train,
    'R2': R2_train,
    'RSE': RSE_train,
    'MSE': MSE_train
}

test_results['Full Model'] = {
    'RSS': RSS_test,
    'R2': R2_test,
    'RSE': RSE_test,
    'MSE': MSE_test
}

# Print the comparison results for single-variable models
print("Single-variable models:")
for var_name in X_train.columns:
    train_stats = train_results[var_name]
    test_stats = test_results[var_name]
    print("\nVariable:", var_name)
    print(f"Training Set - RSS: {train_stats['RSS']:.2f}, R2: {train_stats['R2']:.2f}, RSE: {train_stats['RSE']:.2f}, "
          f"MSE: {train_stats['MSE']:.2f}")
    print(f"Test Set - RSS: {test_stats['RSS']:.2f}, R2: {test_stats['R2']:.2f}, RSE: {test_stats['RSE']:.2f}, MSE: "
          f"{test_stats['MSE']:.2f}")
    print(f"Change in R2: {train_stats['R2'] - test_stats['R2']:.2f}")
    print(f"Change in RSE: {test_stats['RSE'] - train_stats['RSE']:.2f}")

# Print the comparison results for the full multivariate model
print("\nFull multivariate model:")
train_stats = train_results['Full Model']
test_stats = test_results['Full Model']
print(
    f"Training Set - RSS: {train_stats['RSS']:.2f}, R2: {train_stats['R2']:.2f}, RSE: {train_stats['RSE']:.2f}, "
    f"MSE: {train_stats['MSE']:.2f}")
print(
    f"Test Set - RSS: {test_stats['RSS']:.2f}, R2: {test_stats['R2']:.2f}, RSE: {test_stats['RSE']:.2f}, "
    f"MSE: {test_stats['MSE']:.2f}")
print(f"Change in R2: {train_stats['R2'] - test_stats['R2']:.2f}")
print(f"Change in RSE: {test_stats['RSE'] - train_stats['RSE']:.2f}")

# standard imports
import os, random
import numpy as np
import matplotlib.pyplot as plt

# sklearn imports
from sklearn.datasets import *
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn import svm
from PIL import Image

# HW5 Part 1 Step 1
# Function to convert an array of real numbers into an array of 0s and 1s
def binarize(arr, split=10):
    # Calculate the decile thresholds
    percentiles = int(np.ceil(100 / split))
    split_points = np.arange(0, 100 + percentiles, percentiles)
    split_points[split_points > 100] = 100
    deciles = np.percentile(arr, split_points)

    # Create a new array to hold the modified values
    modified_arr = np.zeros_like(arr)

    # Iterate through each decile range and set values accordingly
    for i in range(split):
        print(i)
        if i == split - 1:
            if i % 2 == 0:
                # Set values in even deciles to 0
                modified_arr[(arr >= deciles[i])] = 0
            else:
                # Set values in odd deciles to 1
                modified_arr[(arr >= deciles[i])] = 1
        else:
            if i % 2 == 0:
                # Set values in even deciles to 0
                modified_arr[(arr >= deciles[i]) & (arr < deciles[i + 1])] = 0
            else:
                # Set values in odd deciles to 1
                modified_arr[(arr >= deciles[i]) & (arr < deciles[i + 1])] = 1

    return modified_arr


# Function to generate datasets
def generate_dataset(dataset_type, n_samples=300, noise=0.1, split=10, random_state=0):
    if dataset_type == 'linearly_separable':
        X, y = make_classification(n_samples=n_samples, n_features=2, n_redundant=0, n_informative=2,
                                   random_state=random_state, n_clusters_per_class=1)
    elif dataset_type == 'blobs':
        X, y = make_blobs(n_samples=[n_samples // 2, n_samples // 2], random_state=random_state, cluster_std=noise)
    elif dataset_type == 'quantiles':
        X, y = make_gaussian_quantiles(n_samples=n_samples, n_classes=2, cov=noise, random_state=random_state)
    elif dataset_type == 'moons':
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    elif dataset_type == 'circles':
        X, y = make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=random_state)
    elif dataset_type == 'unstructured':
        X, y = np.random.random(size=(n_samples, 2)), np.random.randint(0, 2, size=(n_samples))
    elif dataset_type == 'swiss':
        X, y = make_swiss_roll(n_samples=n_samples, noise=noise, random_state=random_state)
        X = np.array([X[:, 0], X[:, 2]]).T
        y = binarize(y, split=split)
    elif dataset_type == 'scurve':
        X, y = make_s_curve(n_samples=n_samples, noise=noise, random_state=random_state)
        X = np.array([X[:, 0], X[:, 2]]).T
        y = binarize(y, split=split)
    else:
        raise ValueError("Invalid dataset type")

    X = StandardScaler().fit_transform(X)
    return X, y


# # Generate and visualize data blobs
# Generate the 'blobs' dataset
X, y = generate_dataset('blobs', n_samples=300, noise=0.1, split=10, random_state=42)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train an SVM classifier
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Plot the decision boundary and the data blobs
plt.figure(figsize=(10, 6))
DecisionBoundaryDisplay.from_estimator(clf, X, response_method='predict', cmap=plt.cm.coolwarm, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
plt.title("SVM Decision Boundary with Data Blobs")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")


# I anticipate that this data set will be easy to classify with a linear and non-linear classifier because of the
# simplicity of the blob structure


# Generate and visualize unstructured data
# Generate the 'unstructured' dataset
X, y = generate_dataset('unstructured', n_samples=300, noise=0.1, split=10, random_state=42)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train an SVM classifier
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Plot the decision boundary and the unstructured data
plt.figure(figsize=(10, 6))
DecisionBoundaryDisplay.from_estimator(clf, X, response_method='predict', cmap=plt.cm.coolwarm, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
plt.title("SVM Decision Boundary with Unstructured Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

# I anticipate that this data set will be difficult to classify with a linear classifier due to the unstructured
# nature of the data set, and will not be easier to qualify with a non-linear classifier


# Generate and visualize circles data set
# Generate the 'circles' dataset
X, y = generate_dataset('circles', n_samples=300, noise=0.1, split=10, random_state=42)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train an SVM classifier
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Plot the decision boundary and the circles data
plt.figure(figsize=(10, 6))
DecisionBoundaryDisplay.from_estimator(clf, X, response_method='predict', cmap=plt.cm.coolwarm, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
plt.title("Linear SVM Decision Boundary with Circles Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

# I anticipate that this data set will be difficult to classify with a linear classifier because it takes the shape
# of two circles, which is not linearly separable, however for a non-linear classifier, it should be quite easy to
# separate and classify due to the ability to create complex decision boundaries


# Generate and visualize Gaussian quantiles
# Generate the 'quantiles' dataset
X, y = generate_dataset('quantiles', n_samples=300, noise=0.1, split=10, random_state=42)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train an SVM classifier
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Plot the decision boundary and the Gaussian quantiles data
plt.figure(figsize=(10, 6))
DecisionBoundaryDisplay.from_estimator(clf, X, response_method='predict', cmap=plt.cm.coolwarm, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
plt.title("Linear SVM Decision Boundary with Gaussian Quantiles Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

# I anticipate that this data set will be extremely hard to classify with a linear classifier due to the shape of the
# data distribution, and will be easier to classify with a non-linear classifier due to the ability to create complex
# decision boundaries


# Generate and visualize linearly separable data
# Generate the 'linearly_separable' dataset
X, y = generate_dataset('linearly_separable', n_samples=300, noise=0.1, split=10, random_state=42)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train an SVM classifier
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Plot the decision boundary and the linearly separable data
plt.figure(figsize=(10, 6))
DecisionBoundaryDisplay.from_estimator(clf, X, response_method='predict', cmap=plt.cm.coolwarm, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
plt.title("Linear SVM Decision Boundary with Linearly Separable Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

# I anticipate this data set should be extremely easy to classify with a linear and non-linear classifier because it is
# designed to be linear


# Generate and visualize moons data set
# Generate the 'moons' dataset
X, y = generate_dataset('moons', n_samples=300, noise=0.1, split=10, random_state=42)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train an SVM classifier
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Plot the decision boundary and the moons data for the linear classifier
plt.figure(figsize=(10, 6))
DecisionBoundaryDisplay.from_estimator(clf, X, response_method='predict', cmap=plt.cm.coolwarm, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
plt.title("Linear SVM Decision Boundary with Moons Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

# I anticipate that this data set will be difficult to classify with a linear classifier due to its shape and will be
# easier to classify with a non-linear classifier due to the adaptability of the decision boundary


# Generate and visualize swiss role with 2 split sets
# Generate the 'swiss' dataset
X, y = generate_dataset('swiss', n_samples=300, noise=0.1, split=2, random_state=42)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train an SVM classifier
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Plot the decision boundary and the Swiss roll data for the linear classifier
plt.figure(figsize=(10, 6))
DecisionBoundaryDisplay.from_estimator(clf, X, response_method='predict', cmap=plt.cm.coolwarm, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
plt.title("Linear SVM Decision Boundary with Swiss Roll Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

# I anticipate that this data set will be difficult to classify linearly due to the shape of the data set, and should
# be easier to classify with a non-linear classifier


# Generate and visualize S curve with 2 split sets
# Generate the 'scurve' dataset
X, y = generate_dataset('scurve', n_samples=300, noise=0.1, split=2, random_state=42)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train an SVM classifier
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Plot the decision boundary and the S-curve data
plt.figure(figsize=(10, 6))
DecisionBoundaryDisplay.from_estimator(clf, X, response_method='predict', cmap=plt.cm.coolwarm, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
plt.title("Linear SVM Decision Boundary with S-Curve Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

# I anticipate this data set to be relatively easy to classify with bot a linear and non-linear data set due to the
# style of data visualization, where there should be a relatively linear boundary


# Generate and visualize swiss role with 10 split sets
X, y = generate_dataset('swiss', n_samples=300, noise=0.1, split=10, random_state=42)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train an SVM classifier
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Plot the decision boundary and the Swiss roll data for the linear classifier
plt.figure(figsize=(10, 6))
DecisionBoundaryDisplay.from_estimator(clf, X, response_method='predict', cmap=plt.cm.coolwarm, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
plt.title("Linear SVM Decision Boundary with Swiss Roll Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

# I anticipate this data set to be difficult to classify with both a linear and a non-linear classifier due to the
# location of the splits and the quantity of them


# Generate and visualize S curve with 10 split sets
# Generate the 'scurve' dataset
X, y = generate_dataset('scurve', n_samples=300, noise=0.1, split=10, random_state=42)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train an SVM classifier
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Plot the decision boundary and the S-curve data
plt.figure(figsize=(10, 6))
DecisionBoundaryDisplay.from_estimator(clf, X, response_method='predict', cmap=plt.cm.coolwarm, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
plt.title("Linear SVM Decision Boundary with S-Curve Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

# I anticipate this data set to be extremely difficult to classify with both a linear and non-linear classifier due to
# the quantity of splits

plt.close()


# HW 5 Part 1 Step 2

def kernel_comparison(X, y, support_vectors=True, tight_box=False, if_flag=False):
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    fig = plt.figure(figsize=(15, 5))  # Adjusted the size to fit 4 plots

    for ikernel, kernel in enumerate(['linear', 'poly', 'rbf', 'sigmoid']):
        # Train the SVC
        clf = svm.SVC(kernel=kernel, degree=3, gamma='scale').fit(X_train, y_train)

        # Calculate train and test accuracy
        train_accuracy = accuracy_score(y_train, clf.predict(X_train))
        test_accuracy = accuracy_score(y_test, clf.predict(X_test))

        # Print the accuracies
        print(f"Kernel: {kernel}")
        print(f"Train Accuracy: {train_accuracy:.2f}")
        print(f"Test Accuracy: {test_accuracy:.2f}\n")

        # Settings for plotting
        ax = plt.subplot(1, 4, 1 + ikernel)

        # Plot decision boundary and margins
        common_params = {"estimator": clf, "X": X, "ax": ax}
        DecisionBoundaryDisplay.from_estimator(
            **common_params,
            response_method="predict",
            plot_method="pcolormesh",
            alpha=0.3,
        )
        DecisionBoundaryDisplay.from_estimator(
            **common_params,
            response_method="decision_function",
            plot_method="contour",
            levels=[-1, 0, 1],
            colors=["k", "k", "k"],
            linestyles=["--", "-", "--"],
        )

        if support_vectors:
            # Plot bigger circles around samples that serve as support vectors
            ax.scatter(
                clf.support_vectors_[:, 0],
                clf.support_vectors_[:, 1],
                s=150,
                facecolors="none",
                edgecolors="k",
            )

        # Plot samples by color and add legend
        ax.scatter(X[:, 0], X[:, 1], c=y, s=30, edgecolors="k")
        ax.set_title(kernel)
        ax.axis('off')
        if tight_box:
            ax.set_xlim([X[:, 0].min(), X[:, 0].max()])
            ax.set_ylim([X[:, 1].min(), X[:, 1].max()])

    # plt.show()


# Usage for above datasets
datasets = ['blobs', 'unstructured', 'circles', 'quantiles', 'linearly_separable', 'moons', 'swiss', 'scurve']

for dataset_type in datasets:
    print(f"Kernel comparison for {dataset_type} dataset:")
    X, y = generate_dataset(dataset_type, n_samples=300, noise=0.1, split=2, random_state=42)
    kernel_comparison(X, y)


# To summarize the results of the kernel comparison, I notice that my predictions were accurate for the most part:
# Blobs: accurate guess, easily classified by linear and non-linear
# Unstructured: not easily classified by linear or non-linear
# Circles: accurate, easily classified by non-linear, not by linear
# Quantiles: accurate, easily classified by non-linear, not by linear
# Linearly Separable: accurate guess, fairly easily classified by linear and non-linear
# Moons: accurate, easily classified by non-linear, not by linear
# Swiss: accurate, easily classified by non-linear, not by linear,
# S-curve: accurate, easily classified by both

def kernel_comparison_with_dg(X, y, degrees=[3], gammas=['scale'], support_vectors=True, tight_box=False):
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    fig, axs = plt.subplots(len(degrees) * len(gammas), 4,
                            figsize=(15, 5 * len(degrees) * len(gammas)))  # Adjusted the size to fit plots

    for idegree, degree in enumerate(degrees):
        for igamma, gamma in enumerate(gammas):
            row_idx = idegree * len(gammas) + igamma
            for ikernel, kernel in enumerate(['linear', 'poly', 'rbf', 'sigmoid']):
                # Train the SVC
                if kernel == 'linear':
                    clf = svm.SVC(kernel=kernel).fit(X_train, y_train)
                elif kernel == 'poly':
                    clf = svm.SVC(kernel=kernel, degree=degree, gamma=gamma).fit(X_train, y_train)
                else:
                    clf = svm.SVC(kernel=kernel, gamma=gamma).fit(X_train, y_train)

                # Calculate train and test accuracy
                train_accuracy = accuracy_score(y_train, clf.predict(X_train))
                test_accuracy = accuracy_score(y_test, clf.predict(X_test))

                # Print the accuracies
                print(f"Kernel: {kernel}, Degree: {degree}, Gamma: {gamma}")
                print(f"Train Accuracy: {train_accuracy:.2f}")
                print(f"Test Accuracy: {test_accuracy:.2f}\n")

                # Settings for plotting
                ax = axs[row_idx, ikernel] if len(degrees) * len(gammas) > 1 else axs[ikernel]

                # Plot decision boundary and margins
                common_params = {"estimator": clf, "X": X, "ax": ax}
                DecisionBoundaryDisplay.from_estimator(
                    **common_params,
                    response_method="predict",
                    plot_method="pcolormesh",
                    alpha=0.3,
                )
                DecisionBoundaryDisplay.from_estimator(
                    **common_params,
                    response_method="decision_function",
                    plot_method="contour",
                    levels=[-1, 0, 1],
                    colors=["k", "k", "k"],
                    linestyles=["--", "-", "--"],
                )

                if support_vectors:
                    # Plot bigger circles around samples that serve as support vectors
                    ax.scatter(
                        clf.support_vectors_[:, 0],
                        clf.support_vectors_[:, 1],
                        s=150,
                        facecolors="none",
                        edgecolors="k",
                    )

                # Plot samples by color and add legend
                ax.scatter(X[:, 0], X[:, 1], c=y, s=30, edgecolors="k")
                ax.set_title(f"{kernel}\ndegree={degree}, gamma={gamma}")
                ax.axis('off')
                if tight_box:
                    ax.set_xlim([X[:, 0].min(), X[:, 0].max()])
                    ax.set_ylim([X[:, 1].min(), X[:, 1].max()])

    # plt.show()


# Different degrees and gammas
degrees = [2, 3, 4]
gammas = [0.1, 1, 10]


for dataset_type in datasets:
    print(f"Kernel comparison for {dataset_type} dataset:")
    X, y = generate_dataset(dataset_type, n_samples=300, noise=0.1, split=2, random_state=42)
    kernel_comparison_with_dg(X, y, degrees=degrees, gammas=gammas)


# The degree argument affects polynomial kernel. It changes the model by determining the complexity of the function used
# to create the decision boundary.This affects the model's bias-variance tradeoff by reducing bias but increasing
# variance with a higher degree and reducing variance and increasing bias with a lower degree. As one increases the
# degree, the decision boundary the decision boundary becomes more complex and can fit more intricate patterns, which
# may lead to over fitting. The gamma argument affects rbf and sigmoids. It changes the model by controlling the
# influence of a single training example. This affects the model's bias-variance tradeoff by allowing the model to
# fit the training data more closely with a high gamma (low bias, high variance) or having a smoother decision boundary
# with a low gamma (high bias, low variance). As one increases gamma, the decision boundary of the RBF and sigmoid
# kernels becomes more intricate and can potentially overfit the training data.


# Please see HW5 Pt2 for the flag code!

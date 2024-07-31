# standard imports
import os, random
import numpy as np
import matplotlib.pyplot as plt

# sklearn imports
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from PIL import Image
import heapq


# HW 5 Part 2 step 1
def load_images(folder):
    images = []
    labels = []
    for filename in os.listdir(folder):
        if filename.endswith(".png"):
            img_path = os.path.join(folder, filename)
            img = Image.open(img_path).convert('L')  # Convert image to black and white
            img = np.array(img)//(256/2) # Convert to BW
            images.append(img)
            labels.append(filename.split('.')[0])  # Extract the state code as label
    return images, labels


folder = 'flags'
images, labels = load_images(folder)

# Display four images in a 2x2 grid
fig, axes = plt.subplots(2, 2, figsize=(8, 8))
selected_images = images[:4]  # Select the first 4 images
selected_labels = labels[:4]  # Select the first 4 labels

for i, ax in enumerate(axes.flat):
    if i < len(selected_images):
        ax.imshow(selected_images[i], cmap='gray')
        ax.set_title(selected_labels[i])
    ax.axis('off')
plt.tight_layout()
# plt.show()


def sample_pixels(image, num_samples=100):
    pixel_data = []
    pixel_labels = []
    height, width = image.shape
    for _ in range(num_samples):
        x1 = random.randint(0, width - 1)
        x2 = random.randint(0, height - 1)
        pixel_data.append([x1/width-0.5, x2/width-0.5])
        pixel_labels.append(image[x2,x1])
    return np.array(pixel_data), np.array(pixel_labels, dtype=int)


# Generate synthetic datasets of pixels for the first flag image
first_image = images[0]
pixel_data, pixel_labels = sample_pixels(first_image, num_samples=1000)

# Display the sampled pixel data
plt.figure(figsize=(8, 6))
plt.scatter(pixel_data[:, 0], pixel_data[:, 1], c=pixel_labels, cmap='gray', s=5)
plt.title("Sampled Pixel Data from the Flag Image")
plt.xlabel("x1 (scaled)")
plt.ylabel("x2 (scaled)")
plt.colorbar(label='Pixel Intensity')
# plt.show()


# Visual kernel comparison for selected flags
# Function for kernel comparison
def kernel_comparison(X, y, if_flag=False, support_vectors=True, tight_box=False):

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    if if_flag:
        fig, axs = plt.subplots(2, 2, figsize=(12, 12))
        axes = axs.flatten()
    else:
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))

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
        ax = axes[ikernel]

        if if_flag:
            height, width = int(np.sqrt(len(y))), int(np.sqrt(len(y)))  # Assuming square images for simplicity
            x_min, x_max = -0.5, 0.5
            y_min, y_max = -0.5, 0.5
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
        else:
            x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
            y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1

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

    plt.tight_layout()
    # plt.show()


# Load images, sample pixels, and perform kernel comparison
folder = 'flags'
images, labels = load_images(folder)


# Generate synthetic datasets of pixels for a flag image and perform kernel comparison
for img, label in zip(selected_images, selected_labels):
    X, y = sample_pixels(img, num_samples=1000)
    print(f"Kernel comparison for image: {label}")
    kernel_comparison(X, y, if_flag=True)


print("---------------------\n")


# Non-visual kernel comparison for all flags
def kernel_comparison_quantitative(images, labels):

    # Initialize dictionaries to store results for each kernel
    kernel_results = {kernel: {'best': [], 'worst': []} for kernel in ['linear', 'poly', 'rbf', 'sigmoid']}

    # Store accuracies for each flag and kernel
    flag_accuracies = {label: {kernel: {'train_accuracy': None, 'test_accuracy': None} for kernel in
                               ['linear', 'poly', 'rbf', 'sigmoid']} for label in labels}

    for img, label in zip(images, labels):
        X, y = sample_pixels(img, num_samples=1000)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        for kernel in ['linear', 'poly', 'rbf', 'sigmoid']:
            # Train the SVC
            clf = svm.SVC(kernel=kernel, degree=3, gamma='scale').fit(X_train, y_train)

            # Calculate train and test accuracy
            train_accuracy = accuracy_score(y_train, clf.predict(X_train))
            test_accuracy = accuracy_score(y_test, clf.predict(X_test))

            # Store the accuracies in the dictionary
            flag_accuracies[label][kernel]['train_accuracy'] = train_accuracy
            flag_accuracies[label][kernel]['test_accuracy'] = test_accuracy

            # Record the test accuracy with the flag label for each kernel
            heapq.heappush(kernel_results[kernel]['best'], (test_accuracy, label))
            if len(kernel_results[kernel]['best']) > 3:
                heapq.heappop(kernel_results[kernel]['best'])  # Keep only the top 3

            heapq.heappush(kernel_results[kernel]['worst'], (-test_accuracy, label))
            if len(kernel_results[kernel]['worst']) > 3:
                heapq.heappop(kernel_results[kernel]['worst'])  # Keep only the bottom 3

    # Convert heaps to lists of labels
    for kernel in kernel_results:
        kernel_results[kernel]['best'] = [(label, accuracy) for accuracy, label in kernel_results[kernel]['best']]
        kernel_results[kernel]['worst'] = [(label, -accuracy) for accuracy, label in kernel_results[kernel]['worst']]

    return kernel_results


# Perform kernel comparison and find easiest and hardest flags
kernel_results = kernel_comparison_quantitative(images, labels)

# Print the results
for kernel, results in kernel_results.items():
    print(f"Kernel: {kernel}")
    print("Top 3 best performing datasets:")
    for label, accuracy in results['best']:
        print(f"  Flag: {label}, Test Accuracy: {accuracy:.2f}")
    print("Top 3 worst performing datasets:")
    for label, accuracy in results['worst']:
        print(f"  Flag: {label}, Test Accuracy: {accuracy:.2f}")
    print()


# For these experiments, I set num_samples to 1000 because the results of the experiments seem to be the most stable
# for this number of sampled pixels.
# The linear kernel performed best (i.e., highest test accuracy) on the flags of the following three states:
#
# FL, AL, MD
#
# It performed worst on the flags of the following three states:
#
# RI, AK, NV
#
# The polynomial kernel performed best on the flags of the following three states:
#
# FL, AL, MD
#
# It performed worst on the flags of the following three states:
#
# RI, AK, NV
#
# The radial-basis function kernel performed best on the flags of the following three states:
#
# FL, HI, MD
#
# It performed worst on the flags of the following three states:
#
# RI, NV, AK
#
# The sigmoid kernel performed best on the flags of the following three states:
#
# MD, MO, IA
#
# It performed worst on the flags of the following three states:
#
# RI, AK, NV


# HW 5 Part 2 Step 3

# An arbitrarily complex decision tree would be able to achieve perfect training accuracy on any data set, because
# it has the ability to overfit, so it can grow as complex as needed and can perfectly match training data
#
# For a very large data set of flag pixels, an arbitrarily complex decision tree is likely to achieve (almost) perfect
# test accuracy because a sufficiently large and complex decision tree can capture intricate patterns and relationships
# in the data.
#
# A simple decision tree is likely to perform well on the sampled pixel data of the flags of FL, HI, MD, AL because they
# are unique (and difficult to classify).

# Comparison of SVM and decision tree performance on sampled pixel data for four flags


def compare_svm_decision_tree(images, labels):
    results = {label: {'SVM': {'train_accuracy': None, 'test_accuracy': None},
                       'DecisionTree': {'train_accuracy': None, 'test_accuracy': None}} for label in labels}

    for img, label in zip(images, labels):
        # Generate synthetic data
        X, y = sample_pixels(img, num_samples=1000)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Train and evaluate SVM
        svm_clf = svm.SVC(kernel='rbf', degree=3, gamma='scale').fit(X_train, y_train)
        svm_train_accuracy = accuracy_score(y_train, svm_clf.predict(X_train))
        svm_test_accuracy = accuracy_score(y_test, svm_clf.predict(X_test))

        # Train and evaluate Decision Tree
        dt_clf = DecisionTreeClassifier().fit(X_train, y_train)
        dt_train_accuracy = accuracy_score(y_train, dt_clf.predict(X_train))
        dt_test_accuracy = accuracy_score(y_test, dt_clf.predict(X_test))

        # Store results
        results[label]['SVM']['train_accuracy'] = svm_train_accuracy
        results[label]['SVM']['test_accuracy'] = svm_test_accuracy
        results[label]['DecisionTree']['train_accuracy'] = dt_train_accuracy
        results[label]['DecisionTree']['test_accuracy'] = dt_test_accuracy

    return results


# Compare SVM and Decision Tree performance
comparison_results = compare_svm_decision_tree(selected_images, selected_labels)

# Print results
for flag, metrics in comparison_results.items():
    print(f"Flag: {flag}")
    print(f"  SVM Train Accuracy: {metrics['SVM']['train_accuracy']:.2f}")
    print(f"  SVM Test Accuracy: {metrics['SVM']['test_accuracy']:.2f}")
    print(f"  Decision Tree Train Accuracy: {metrics['DecisionTree']['train_accuracy']:.2f}")
    print(f"  Decision Tree Test Accuracy: {metrics['DecisionTree']['test_accuracy']:.2f}")

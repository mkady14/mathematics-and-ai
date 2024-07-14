import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os, time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import seaborn as sns

# HW3 Step 1
# toggle settings
add_noise = False

# Initialize lists for image collection
train_images = []
test_images = []

for i, images in enumerate([train_images, test_images]):

    # set paths to images of pandas and bears in train and test set
    datasetname = ['Train', 'Test'][i]
    folder_path1 = 'PandasBears/{}/Pandas/'.format(datasetname)
    folder_path2 = 'PandasBears/{}/Bears/'.format(datasetname)

    for folder_path in [folder_path1, folder_path2]:

        # print the name of the folder that is currently being processed
        print(folder_path, end=' ')

        # go through all files in the folder
        file_count = 0
        for filename in os.listdir(folder_path):

            # find the files that are JPEGs
            if filename.endswith(".jpg") or filename.endswith(".jpeg"):

                # add 1 to the file count
                file_count += 1

                # Construct full file path
                file_path = os.path.join(folder_path, filename)

                # import image
                image = plt.imread(file_path, format='jpeg')

                # convert to gray scale
                image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])

                # decrease image size by 50%
                image = image[::2, ::2]

                if add_noise:
                    # add some noise
                    image = image + np.random.normal(scale=100, size=image.shape)

                # add the new image to collection
                images.append(image)

        print('has {} images'.format(file_count))

# look at 4 random bears
for i0, i in enumerate(np.random.randint(0, 500, size=4)):
    plt.subplot(2,2,1+i0)
    plt.imshow(train_images[i][::2,::2],cmap='Greys_r')


# HW3 Step 2
# Flatten train images and form matrix A
A = np.array([img.flatten() for img in train_images])

# Compute the mean image of the train set
mean_image = np.mean(A, axis=0)

# Center the matrix A by subtracting the mean image
A_centered = A - mean_image

# And transpose it to the correct shape
A_T = A_centered.T

# Perform SVD
U, S, Vh = np.linalg.svd(A_T, full_matrices=False)

# show the first four eigenbears
for i in range(4):
    plt.subplot(2,2,1+i)
    plt.imshow((U[:, i]).reshape((128,128)), cmap='Greys_r',
        # force colormap to be the same for all four
        vmin=-np.max(np.abs(U[:,:4])),
        vmax=np.max(np.abs(U[:,:4])))
    plt.colorbar()
plt.subplots_adjust(wspace=0.4)

# Clustering of panda bears and brown bears along the first and second principal component
# indices of pandas in the test set
indices_pandas = range(50)
# indices of brown bears in the test set
indices_brownbears = range(50,100)

for i, indices in enumerate([indices_pandas, indices_brownbears]):
    # get projections of data onto principal component 1
    p1 = [np.dot(U[:,0],np.ravel(test_images[x])) for x in indices]
    # get projections of data onto principal component 2
    p2 = [np.dot(U[:,1],np.ravel(test_images[x])) for x in indices]
    plt.plot(p1, p2, marker='+x'[i], lw=0, label=['Pandas', 'Grizzlies'][i])

# annotate axes
plt.xlabel('Principal component 1')
plt.ylabel('Principal component 2')
# add legend
plt.legend()


# HW3 Step 3
# construct response variable: Train set was created by appending 250 pandas
# and THEN 250 brown bears to the list of training images. We code pandas as
# '0' and brown bears as '1'.
y_train =  np.concatenate([np.zeros(250), np.ones(250)])

# Test set was created by appending 50 pandas and THEN 50 brown bears to the
# list of test images. We code pandas as '0' and brown bears as '1'.
y_test = np.concatenate([np.zeros(50), np.ones(50)])

print('   k\t|  # errors\t| misclassified bears')
print('--------------------------------------------')
for k in range(1,16):
    # fit KNN model
    modelKN = KNeighborsClassifier(n_neighbors=k).fit(A_T.T, y_train)
    # use model to make predictions on the test set
    predictions = [modelKN.predict([np.ravel(test_images[i])]) for i in range(len(y_test))]
    # detect misclassifications
    errors = np.abs((np.array(predictions).T)[0]-y_test)
    # print results to table
    print('    {}\t|      {} \t| {}'.format(k, int(np.sum(errors)), (np.argwhere(errors).T)[0]))


# Lists to store accuracy scores for each k
accuracy_scores = []

print('      k\t|      Accuracy')
print('-------------------------')
for k in range(1,16):
    # Fit KNN model
    modelKN = KNeighborsClassifier(n_neighbors=k).fit(A_T.T, y_train)

    # Use model to make predictions on the test set
    predictions = [modelKN.predict([np.ravel(test_images[i])])[0] for i in range(len(y_test))]

    # Compute accuracy score
    accuracy = accuracy_score(y_test, predictions)
    accuracy_scores.append(accuracy)

    # Print results to table
    print('    {}\t|   {:.2f}%'.format(k, accuracy * 100))

    # Find the best k based on highest accuracy score
best_k = np.argmax(accuracy_scores) + 1  # +1 because k starts from 1
print('\nBest k value:', best_k)


# Initialize lists to store misclassified images and their indices
misclassified_indices = []
misclassified_images = []

# Iterate through each test image
for i in range(len(y_test)):
    # Predict the label for the current test image
    prediction = modelKN.predict([np.ravel(test_images[i])])[0]

    # Compare the predicted label with the true label
    if prediction != y_test[i]:
        # If they are different, the image is misclassified
        misclassified_indices.append(i)
        misclassified_images.append(test_images[i])

# Print the indices of misclassified bears
print("Indices of misclassified bears:", misclassified_indices)

# Display the misclassified bear images
plt.figure(figsize=(10, 6))
for i, index in enumerate(misclassified_indices):
    plt.subplot(2, len(misclassified_indices) // 2 + 1, i + 1)
    plt.imshow(test_images[index], cmap='gray')
    plt.title(
        'Predicted: {}\nActual: {}'.format(int(modelKN.predict([np.ravel(test_images[index])])[0]), int(y_test[index])))
    plt.axis('off')

plt.tight_layout()

# These bears might be hard to classify because of the similarities between pandas and bears in gray scale


# HW3 Step 4

# Fit Logistic Regression model

# Train Logistic Regression model
logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(A_T.T, y_train)

# Make predictions on the test set
logistic_predictions = logistic_model.predict([np.ravel(img) for img in test_images])

# Compute accuracy
logistic_accuracy = accuracy_score(y_test, logistic_predictions)
print('Logistic Regression Accuracy: {:.2f}%'.format(logistic_accuracy * 100))

# Confusion matrix for Logistic Regression
logistic_cm = confusion_matrix(y_test, logistic_predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(logistic_cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Panda', 'Bear'], yticklabels=['Panda', 'Bear'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Logistic Regression Confusion Matrix')


# Fit LDA Model

# Train LDA model
lda_model = LinearDiscriminantAnalysis()
lda_model.fit(A_T.T, y_train)

# Make predictions on the test set
lda_predictions = lda_model.predict([np.ravel(img) for img in test_images])

# Compute accuracy
lda_accuracy = accuracy_score(y_test, lda_predictions)
print('LDA Accuracy: {:.2f}%'.format(lda_accuracy * 100))

# Confusion matrix for LDA
lda_cm = confusion_matrix(y_test, lda_predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(lda_cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Panda', 'Bear'], yticklabels=['Panda', 'Bear'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('LDA Confusion Matrix')

# Find the best k value from previous code
best_k = 6  # Assume best_k is determined from previous steps

# The Logistic Regression provides a better test accuracy

# Train KNN model with the best k
knn_model = KNeighborsClassifier(n_neighbors=best_k)
knn_model.fit(A_T.T, y_train)

# Make predictions on the test set
knn_predictions = knn_model.predict([np.ravel(img) for img in test_images])

# Compute accuracy
knn_accuracy = accuracy_score(y_test, knn_predictions)
print('KNN Accuracy (k={}): {:.2f}%'.format(best_k, knn_accuracy * 100))

# The non-parametric classification provides the best test fit

# Construct the bear mask:
plt.imshow(np.abs((logistic_model.coef_).reshape((128, 128))))
plt.colorbar()
plt.show()

# I notice that this bear mask is terrifying


# HW3 Step 5

# With add noise set to true, the best K value is still 6, which is still the best test-accuracy compared to LDA and
# Logistic regression. However, all the Accuracy values have changed to be slightly lower than they were when
# add noise was false.

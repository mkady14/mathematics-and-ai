# import packages
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
import time
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns

# Results from rerunning example.ipynb with Adam:
# Epoch [1/10], Train Loss: 1.2916, Validation Loss: 0.4264, Train Acc: 59.67%, Val Acc: 89.08%
# Epoch [2/10], Train Loss: 0.2925, Validation Loss: 0.2264, Train Acc: 91.94%, Val Acc: 93.25%
# Epoch [3/10], Train Loss: 0.1820, Validation Loss: 0.1664, Train Acc: 94.78%, Val Acc: 95.18%
# Epoch [4/10], Train Loss: 0.1361, Validation Loss: 0.1276, Train Acc: 96.03%, Val Acc: 96.21%
# Epoch [5/10], Train Loss: 0.1099, Validation Loss: 0.1098, Train Acc: 96.74%, Val Acc: 96.71%
# Epoch [6/10], Train Loss: 0.0939, Validation Loss: 0.0971, Train Acc: 97.19%, Val Acc: 97.09%
# Epoch [7/10], Train Loss: 0.0808, Validation Loss: 0.0870, Train Acc: 97.57%, Val Acc: 97.46%
# Epoch [8/10], Train Loss: 0.0733, Validation Loss: 0.0813, Train Acc: 97.84%, Val Acc: 97.52%
# Epoch [9/10], Train Loss: 0.0653, Validation Loss: 0.0763, Train Acc: 98.05%, Val Acc: 97.72%
# Epoch [10/10], Train Loss: 0.0587, Validation Loss: 0.0694, Train Acc: 98.29%, Val Acc: 97.81%
# Adam Test Accuracy: 98.17%


# Load and preprocess the MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# Download and load the training and test datasets
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Split the training dataset into a training set and a validation set
train_set, val_set = random_split(train_dataset, [50000, 10000])

# Create data loaders for the training, validation, and test sets
train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
val_loader = DataLoader(val_set, batch_size=128, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)


# Define the new CNN architecture
class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        # modified from 32 to 15 per convolution testing
        self.conv1 = nn.Conv2d(1, 15, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(15, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # No padding in the final pooling layer

        self.fc1 = nn.Linear(128 * 3 * 3, 512)  # Adjusted for the final feature map size
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(-1, 128 * 3 * 3)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Define training pipeline including validation after each epoch
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=10):
    train_loss, val_loss = [], []
    train_acc, val_acc = [], []

    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss.append(running_loss / len(train_loader))
        train_acc.append(100 * correct / total)

        model.eval()
        val_running_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss.append(val_running_loss / len(val_loader))
        val_acc.append(100 * val_correct / val_total)

        print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {running_loss / len(train_loader):.4f}, '
              f'Validation Loss: {val_running_loss / len(val_loader):.4f}, '
              f'Train Acc: {100 * correct / total:.2f}%, Val Acc: {100 * val_correct / val_total:.2f}%')

    return train_loss, val_loss, train_acc, val_acc


# Build and train a model
model = CustomCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Measure the training time
start_time = time.time()
train_loss, val_loss, train_acc, val_acc = train_model(model, train_loader, val_loader, criterion, optimizer, epochs=5)
end_time = time.time()

training_time = end_time - start_time
print(f"Training Time: {training_time:.2f} seconds")


# Plot training & validation accuracy/loss
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(train_loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')

plt.show()

# Evaluate the model on test set
model.eval()
test_correct, test_total = 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()

test_acc = 100 * test_correct / test_total
print(f'Test Accuracy: {test_acc:.2f}%')

# The time that it took for the model to run greatly increased, however the validation accuracy also increased

# Results with Epochs = 10
# Epoch [1/10], Train Loss: 0.2192, Validation Loss: 0.0625, Train Acc: 93.11%, Val Acc: 98.13%
# Epoch [2/10], Train Loss: 0.0491, Validation Loss: 0.0490, Train Acc: 98.44%, Val Acc: 98.42%
# Epoch [3/10], Train Loss: 0.0342, Validation Loss: 0.0499, Train Acc: 98.94%, Val Acc: 98.43%
# Epoch [4/10], Train Loss: 0.0246, Validation Loss: 0.0550, Train Acc: 99.19%, Val Acc: 98.46%
# Epoch [5/10], Train Loss: 0.0181, Validation Loss: 0.0375, Train Acc: 99.42%, Val Acc: 99.00%
# Epoch [6/10], Train Loss: 0.0148, Validation Loss: 0.0427, Train Acc: 99.50%, Val Acc: 98.86%
# Epoch [7/10], Train Loss: 0.0142, Validation Loss: 0.0431, Train Acc: 99.53%, Val Acc: 99.00%
# Epoch [8/10], Train Loss: 0.0110, Validation Loss: 0.0470, Train Acc: 99.62%, Val Acc: 98.87%
# Epoch [9/10], Train Loss: 0.0112, Validation Loss: 0.0524, Train Acc: 99.60%, Val Acc: 98.89%
# Epoch [10/10], Train Loss: 0.0065, Validation Loss: 0.0589, Train Acc: 99.80%, Val Acc: 98.82%
# Training Time: 259.98 seconds
# Test Accuracy: 98.97%


# Results with Epochs = 5
# Epoch [1/5], Train Loss: 0.2010, Validation Loss: 0.0601, Train Acc: 93.85%, Val Acc: 98.07%
# Epoch [2/5], Train Loss: 0.0438, Validation Loss: 0.0463, Train Acc: 98.63%, Val Acc: 98.48%
# Epoch [3/5], Train Loss: 0.0306, Validation Loss: 0.0359, Train Acc: 99.00%, Val Acc: 98.75%
# Epoch [4/5], Train Loss: 0.0243, Validation Loss: 0.0304, Train Acc: 99.25%, Val Acc: 99.05%
# Epoch [5/5], Train Loss: 0.0173, Validation Loss: 0.0322, Train Acc: 99.47%, Val Acc: 99.01%
# Training Time: 145.38 seconds
# Test Accuracy: 98.88%
# The training time greatly decreases without impacting the Test Accuracy, making this an optimal value for k


# # Testing different convolution layers:
# class CustomCNN2(nn.Module):
#     def __init__(self, c):
#         super(CustomCNN2, self).__init__()
#         self.conv1 = nn.Conv2d(1, c, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(c, 64, kernel_size=3, padding=1)
#         self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
#         self.activation = nn.ReLU()
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.fc1 = nn.Linear(128 * 3 * 3, 512)
#         self.fc2 = nn.Linear(512, 10)
#
#     def forward(self, x):
#         x = self.activation(self.conv1(x))
#         x = self.pool(x)
#         x = self.activation(self.conv2(x))
#         x = self.pool(x)
#         x = self.activation(self.conv3(x))
#         x = self.pool(x)
#         x = x.view(-1, 128 * 3 * 3)
#         x = self.activation(self.fc1(x))
#         x = self.fc2(x)
#         return x
#
# def find_best_c():
#     best_c = None
#     best_val_acc = 0
#     results = {}
#
#     for c in range(2, 16):
#         print(f"\nTraining with c = {c} channels in the first convolutional layer.")
#
#         model = CustomCNN2(c)
#         criterion = nn.CrossEntropyLoss()
#         optimizer = optim.Adam(model.parameters(), lr=0.001)
#
#         start_time = time.time()
#         train_loss, val_loss, train_acc, val_acc = train_model(model, train_loader, val_loader, criterion, optimizer,
#                                                                epochs=5)
#         end_time = time.time()
#
#         val_acc_last = val_acc[-1]  # Get the validation accuracy of the last epoch
#         results[c] = val_acc_last
#
#         print(f"Validation Accuracy for c = {c}: {val_acc_last:.2f}%")
#         print(f"Training Time: {end_time - start_time:.2f} seconds")
#
#         if val_acc_last > best_val_acc:
#             best_val_acc = val_acc_last
#             best_c = c
#
#     print(f"\nBest c = {best_c} with Validation Accuracy = {best_val_acc:.2f}%")
#     return best_c, results
#
#
# # Run the search for the best c
# best_c, results = find_best_c()
#
# # Update the model architecture with the best c
# model = CustomCNN(best_c)

# The best value for c = 15. Please do not run the above code for more than one c value, it takes FOREVER to test them

# Function to visualize the feature maps produced by different layers for a given image
# def plot_mapped_features(model, image, layers):
#     '''Example usage:
#
#     # >>> examples = iter(test_loader)
#     # >>> example_data, example_labels = next(examples) # get one batch from test set
#     # >>> example_image = example_data[0]
#     # >>> layers = [model.conv1, model.pool, model.conv2, model.pool]
#     # >>> plot_mapped_features(model, example_image, layers)
#
#     '''
#     # Add a batch dimension to the image tensor (from (channels, height, width) to (1, channels, height, width))
#     x = image.unsqueeze(0)
#     # Create a subplot with 1 row and len(layers) columns
#     fig, axes = plt.subplots(1, len(layers))
#     # Iterate over the specified layers
#     for i, layer in enumerate(layers):
#         # Pass the image through the current layer
#         x = layer(x)
#         # Detach the feature map from the computation graph and move it to CPU, then convert it to a NumPy array
#         # Visualize the first channel of the feature map
#         axes[i].imshow(x[0, 0].detach().cpu().numpy(), cmap='gray')
#         # Turn off the axis for a cleaner look
#         axes[i].axis('off')
#     # Display the feature maps
#     plt.show()


def plot_mapped_features(model, image, layers):
    '''Example usage:

    >>> examples = iter(test_loader)
    >>> example_data, example_labels = next(examples) # get one batch from test set
    >>> example_image = example_data[0]
    >>> layers = [model.conv1, model.pool, model.conv2, model.pool, model.conv3, model.pool]
    >>> plot_mapped_features(model, example_image, layers)

    '''
    # Add a batch dimension to the image tensor (from (channels, height, width) to (1, channels, height, width))
    x = image.unsqueeze(0)

    for i, layer in enumerate(layers):
        # Pass the image through the current layer
        x = layer(x)

        # Get the number of channels in the current layer
        num_channels = x.size(1)

        # Create a grid to display all channels for the current layer
        fig, axes = plt.subplots(1, num_channels, figsize=(15, 15))
        fig.suptitle(f'Layer {i + 1}: {str(layer)}')

        for j in range(num_channels):
            # Visualize the j-th channel of the feature map
            axes[j].imshow(x[0, j].detach().cpu().numpy(), cmap='gray')
            axes[j].axis('off')

        plt.show()

# Some changes that I observed between the channels was the focus between different images and the differences in the
# images themselves. For example, channel 1 had a lot fewer pixels than channel 2, even though they came from the same
# layer.


# Function to visualize the filters of a given convolutional layer
def plot_filters(layer, n_filters=6):
    '''Example usage:

    >>> layer = model.conv1
    >>> plot_filters(layer, n_filters=6)

    '''
    # Clone the weights of the convolutional layer to avoid modifying the original weights
    filters = layer.weight.data.clone()
    # Normalize the filter values to the range [0, 1] for better visualization
    filters = filters - filters.min()
    filters = filters / filters.max()
    # Select the first n_filters to visualize
    filters = filters[:n_filters]
    # Create a subplot with 1 row and n_filters columns
    fig, axes = plt.subplots(1, n_filters)
    # Iterate over the selected filters
    for i, filter in enumerate(filters):
        # Transpose the filter dimensions to (height, width, channels) for visualization
        axes[i].imshow(np.transpose(filter, (1, 2, 0)))
        # Turn off the axis for a cleaner look
        axes[i].axis('off')
    # Display the filters
    plt.show()

# Train log regression :(


# def dataloader_to_numpy(dataloader):
#     X, y = [], []
#     for images, labels in dataloader:
#         X.append(images.view(images.size(0), -1).numpy())  # Flatten the images
#         y.append(labels.numpy())
#     X = np.concatenate(X, axis=0)
#     y = np.concatenate(y, axis=0)
#     return X, y
#
# # Convert the training, validation, and test sets to NumPy arrays
# X_train, y_train = dataloader_to_numpy(train_loader)
# X_val, y_val = dataloader_to_numpy(val_loader)
# X_test, y_test = dataloader_to_numpy(test_loader)
#
# # Train the logistic regression model using the training set
# logistic_model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
# logistic_model.fit(X_train, y_train)
#
# # Predict on the validation set
# y_val_pred = logistic_model.predict(X_val)
#
# # Print accuracy score for validation set
# val_accuracy = accuracy_score(y_val, y_val_pred)
# print(f'Validation Accuracy: {val_accuracy:.4f}')
#
# # Print classification report for validation set
# print(classification_report(y_val, y_val_pred))
#
# # Predict on the test set
# y_test_pred = logistic_model.predict(X_test)
#
# # Print accuracy score for test set
# test_accuracy = accuracy_score(y_test, y_test_pred)
# print(f'Test Accuracy: {test_accuracy:.4f}')
#
# # Print classification report for test set
# print(classification_report(y_test, y_test_pred))
#
# # Display confusion matrix for test set
# conf_matrix = confusion_matrix(y_test, y_test_pred)
# plt.figure(figsize=(10, 7))
# plt.imshow(conf_matrix, cmap='Blues')
# plt.title('Confusion Matrix')
# plt.colorbar()
# plt.show()


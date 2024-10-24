import numpy as np
import matplotlib.pyplot as plt
from keras import models
import random

# Step 1: Load the pre-trained model
model = models.load_model('mnist_cnn_model.h5')

# Step 2: Load the dataset (correct the number of images to match the labels)
data = np.load("mnist_compressed.npz")
X_test, y_test = data['test_images'][:10000], data['test_labels']  # Use the first 10,000 images

# Verify shapes of training and test sets
print(f"X_train shape: {X_test.shape}, y_train shape: {y_test.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

# Step 3: Preprocess the data
X_test = X_test.astype('float32') / 255.0  # Normalize the pixel values
X_test = X_test.reshape(-1, 28, 56, 1)     # Reshape for the model (28, 28, 1)

# Step 4: Select 3 random images from the dataset for prediction
random_indices = random.sample(range(10000), 3)  # Adjust the range to be within the dataset size
random_images = X_test[random_indices]
random_labels = y_test[random_indices]

# Step 5: Make predictions on the selected images
predictions = model.predict(random_images)

# Step 6: Display each image with its predicted label
for i in range(3):
    plt.imshow(random_images[i].reshape(28, 56), cmap='gray')  # Display the image
    plt.title(f"True label: {random_labels[i]}, Predicted: {np.argmax(predictions[i])}")
    plt.axis('off')  # Turn off axis display
    plt.show()  # Show the plot

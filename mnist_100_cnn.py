import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

from keras import utils
from keras import models
from keras import layers

# Load data
data = np.load("mnist_compressed.npz")
# Load data (correcting the order)
X_train, y_train, X_test, y_test = data['train_images'], data['train_labels'], data['test_images'], data['test_labels']

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")


# Reshape the data: Adjust for (28, 56) image shape, 1 color channel (grayscale)
X_train = X_train.reshape(-1, 28, 56, 1)
X_test = X_test.reshape(-1, 28, 56, 1)

# Normalize the pixel values (0-255) to (0-1)
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Convert labels to one-hot encoding for 100 classes
y_train = utils.to_categorical(y_train, 100)
y_test = utils.to_categorical(y_test, 100)

# Create the CNN model
model = models.Sequential()

# First convolutional layer: Adjust input_shape for (28, 56, 1)
model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 56, 1)))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Dropout(0.25))

# Second convolutional layer
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Dropout(0.25))

# Flatten the output from the convolutional layers
model.add(layers.Flatten())

# Fully connected layers
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))

# Output layer for 100 classes
model.add(layers.Dense(100, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, batch_size=64, epochs=10, validation_data=(X_test, y_test))

# Save the entire model to a file
model.save('mnist_cnn_model.h5')

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

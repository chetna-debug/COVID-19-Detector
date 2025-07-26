import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

# Image size
IMG_SIZE = 100

# Load and preprocess images
def load_images_from_folder(folder_path, label):
    images = []
    labels = []
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        try:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                images.append(img)
                labels.append(label)
        except Exception as e:
            print(f"[ERROR] Could not read image: {img_path}, {e}")
    return images, labels

print("[INFO] Loading images...")

# Training data
train_covid_images, train_covid_labels = load_images_from_folder(r'C:\Users\cheth\Downloads\archive (3)\Dataset\Train\Covid', 0)
train_normal_images, train_normal_labels = load_images_from_folder(r'C:\Users\cheth\Downloads\archive (3)\Dataset\Train\Normal', 1)

# Validation data
val_covid_images, val_covid_labels = load_images_from_folder(r'C:\Users\cheth\Downloads\archive (3)\Dataset\Val\Covid', 0)
val_normal_images, val_normal_labels = load_images_from_folder(r'C:\Users\cheth\Downloads\archive (3)\Dataset\Val\Normal', 1)

# Combine datasets
X = np.array(train_covid_images + train_normal_images + val_covid_images + val_normal_images)
y = np.array(train_covid_labels + train_normal_labels + val_covid_labels + val_normal_labels)

# Normalize images
X = X / 255.0

# Reshape images to include channel dimension
X = X.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

# One-hot encode labels
from tensorflow.keras.utils import to_categorical
y = to_categorical(y, 2)

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the CNN model
model = Sequential([
    Conv2D(64, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(2, activation='softmax')
])

model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=3)

print("[INFO] Training model...")
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, callbacks=[early_stop])

# Save model
model.save("model.h5")
print("[INFO] Model saved as model.h5")

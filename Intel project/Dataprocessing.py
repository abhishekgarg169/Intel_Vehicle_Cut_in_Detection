import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    resized_image = cv2.resize(image, (224, 224))
    normalized_image = resized_image / 255.0
    return normalized_image

def load_dataset(data_dir):
    images = []
    labels = []
    for label in ['cutin', 'nocutin']:
        label_dir = os.path.join(data_dir, label)
        for image_name in os.listdir(label_dir):
            image_path = os.path.join(label_dir, image_name)
            image = preprocess_image(image_path)
            images.append(image)
            labels.append(1 if label == 'cutin' else 0)
    return np.array(images), np.array(labels)

# Load and preprocess data
data_dir = 'path/to/data'
images, labels = load_dataset(data_dir)

# Split data into training, validation, and test sets
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)
train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)

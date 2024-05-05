import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Function to load and preprocess images
def load_and_preprocess_images(dataset_directory, width, height):
    images = []
    labels = []
    for label in os.listdir(dataset_directory):
        label_dir = os.path.join(dataset_directory, label)
        for filename in os.listdir(label_dir):
            img_path = os.path.join(label_dir, filename)
            image = cv2.imread(img_path)
            if image is not None:
                # Convert image to grayscale
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                # Resize image to a fixed size
                resized_image = cv2.resize(gray, (width, height))
                # Flatten image array
                images.append(resized_image.flatten())
                # Assign label (0 for fake, 1 for real)
                labels.append(1 if label == 'real' else 0)
    return np.array(images), np.array(labels)

# Load and preprocess images
dataset_directory = "C:/Users/HP/Desktop/real_currency/currency_dataset"
width = 100  # Adjust the width as needed
height = 100  # Adjust the height as needed
images, labels = load_and_preprocess_images(dataset_directory, width, height)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Train a classifier
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)

# Evaluate the classifier
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Now you can use this trained model for real-time currency recognition
# For example, you can use classify a new image like this:
def classify_currency(image_path, model, width, height):
    # Load and preprocess the image
    image = cv2.imread(image_path)
    if image is not None:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized_image = cv2.resize(gray, (width, height))
        flattened_image = resized_image.flatten()
        # Predict using the model
        prediction = model.predict([flattened_image])
        if prediction[0] == 1:
            print("Real currency")
        else:
            print("Fake currency")
    else:
        print("Unable to load image")

# Example usage:
image_path = "C:/Users/HP/Desktop/real_currency/test_1.jpg"
classify_currency(image_path, clf, width, height)

# import tensorflow as tf
from tensorflow import keras
# from tensorflow.keras import layers
import numpy as np
from PIL import Image

from image_classifier import train_generator

# Load the trained model
model = keras.models.load_model('saved_models/image_classifier.keras')

# Define constants
img_height, img_width = 128, 128


# Preprocess the user-provided image for prediction
def preprocess_image(image_path):
    img = Image.open(image_path).resize((img_width, img_height))
    img_array = np.array(img) / 255.0  # Normalize pixel values to be between 0 and 1
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array


# Get user input for the image path
user_image_path = input("Enter the path to the image you want to classify: ")

# Preprocess the user-provided image
user_image = preprocess_image(user_image_path)

# Make a prediction using the trained model
predictions = model.predict(user_image)

# Get the predicted class index
predicted_class_index = np.argmax(predictions[0])

# Map the class index to the actual class name using the class indices from the training generator
class_indices = train_generator.class_indices
predicted_class_name = [k for k, v in class_indices.items() if v == predicted_class_index][0]

# Print the predicted class name
print("Predicted Dog Breed: ", predicted_class_name)

# Dog Breed Classifier

## Overview
This project implements a dog breed classifier using a convolutional neural network (CNN) with TensorFlow and Keras. The model is trained to classify dog breeds based on a dataset of images.

## Project Structure
- **image_classifier.py**: The main script that contains the code for model training and prediction.
- **saved_models**: Directory to save the trained model (`model.keras`).
- **dog-breeds**
  - **train**: Training set directory.
  - **valid**: Validation set directory.
  - **test**: Testing set directory.

## Setup and Dependencies
1. Install the required dependencies:
   ```bash
   pip install tensorflow pillow
   ```
2. Run the script:
   ```bash
   python image_classifier.py
   ```

## Usage
1. Train the model by running `image_classifier.py`. Adjust the script to point to your dataset.
2. Use the trained model to make predictions on dog breed types. You can input a new image path when prompted.

## Important Files
- **image_classifier.py**: Main script for model training and prediction.
- **model/model.keras**: Trained model saved in keras format.
- **dog-breeds**: Directory containing the dataset.

## Note
- The dataset should be organized with subdirectories for each class in both the training and testing sets.
- Modify the script paths to match the location of your dataset.

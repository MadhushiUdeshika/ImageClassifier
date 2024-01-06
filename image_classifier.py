# import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Define constants
batch_size = 32
img_height, img_width = 128, 128
epochs = 10

# Create data generators for data augmentation and normalization
train_data = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values to be between 0 and 1
    shear_range=0.2,  # Shear transformations
    zoom_range=0.2,   # Zoom transformations
    horizontal_flip=True,  # Horizontal flipping
)

valid_data = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)  # Only rescale for validation
test_data = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)  # Only rescale for testing

# Define data paths
train_path = 'dog-breeds/train'
valid_path = 'dog-breeds/valid'
test_path = 'dog-breeds/test'

# Create data generators for training, validation, and testing
train_generator = train_data.flow_from_directory(
    train_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',  # Use categorical mode for multi-class classification
)

valid_generator = valid_data.flow_from_directory(
    valid_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
)

test_generator = test_data.flow_from_directory(
    test_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
)

# Build the model
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(70, activation='softmax'),
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',  # Categorical crossentropy for multi-class classification
    metrics=['accuracy'],
)

# Train the model
model.fit(
    train_generator,
    epochs=epochs,
    validation_data=valid_generator,  # Use a validation set during training to monitor performance
)
model.save('saved_models/image_classifier.keras')


from tensorflow.keras.preprocessing import image
import numpy as np
from IPython.display import SVG, Image
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import BatchNormalization, Activation, MaxPooling2D
from tensorflow.keras.layers import Dense, Input, Flatten, Dropout, Conv2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import matplotlib.pyplot as plt
import os

# Load the saved model
classifier = tf.keras.models.load_model('my_disease.h5')

img_size = 48
batch_size = 64

# Define the data generators for training and validation
datagen_train = ImageDataGenerator(horizontal_flip=True)
train_generator = datagen_train.flow_from_directory(
    r"C:\Users\prave\OneDrive\Documents\plant_disease_website\dataset\train",
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True)

datagen_validation = ImageDataGenerator(horizontal_flip=True)
validation_generator = datagen_validation.flow_from_directory(
    r"C:\Users\prave\OneDrive\Documents\plant_disease_website\dataset\test",
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True)

# Load a test image and predict its class
path = r"C:\Users\prave\Downloads\portfolios\Portfolio_trail_2\images\vattakay.jpg"
test_image = image.load_img(path)
plt.imshow(test_image)

test_img = image.load_img(path, target_size=(img_size, img_size))
test_img = image.img_to_array(test_img)
test_img = np.expand_dims(test_img, axis=0)
result = classifier.predict(test_img)
a = result.argmax()
s = train_generator.class_indices
name = []
for i in s:
    name.append(i)
for i in range(len(s)):
    if i == a:
        p = name[i]
print("Predicted class:", p)
plt.imshow(test_image)

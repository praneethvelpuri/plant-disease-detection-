from matplotlib.pyplot import imshow
import numpy as np
from keras.models import load_model
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Input, Flatten, Dropout, Conv2D
from keras.layers import BatchNormalization, Activation, MaxPooling2D
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.utils import plot_model
from IPython.display import SVG, Image
img_size = 48
batch_size = 64
datagen_train = ImageDataGenerator(horizontal_flip=True)
train_generator = \
    datagen_train.flow_from_directory(r"C:\Users\prave\Plant Disease Detection\plant-disease-detection\dataset\train",
                                      target_size=(48, 48), batch_size=batch_size, class_mode='categorical', shuffle=True)
datagen_validation = ImageDataGenerator(horizontal_flip=True)
validation_generator = \
    datagen_train.flow_from_directory(r"C:\Users\prave\Plant Disease Detection\plant-disease-detection\dataset\train",
                                      target_size=(48, 48), batch_size=batch_size, class_mode='categorical', shuffle=True)
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    r"C:\Users\prave\Plant Disease Detection\plant-disease-detection\dataset\test",
    target_size=(48, 48),
    batch_size=32,
    class_mode='categorical')
# initialising CNN

# Create the model
model = Sequential()

# conv-1
model.add(Conv2D(64, (3, 3), padding='same', input_shape=(48, 48, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# conv-2
model.add(Conv2D(128, (5, 5), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# conv-3
model.add(Conv2D(512, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# conv-4
model.add(Conv2D(512, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(38, activation='softmax'))

# Compile the model
opt = Adam(learning_rate=0.0005)
model.compile(optimizer=opt, loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()
# Define the training parameters
epochs = 10
steps_per_epoch = train_generator.n // train_generator.batch_size
validation_steps = validation_generator.n // validation_generator.batch_size

# Define the callbacks
checkpoint = ModelCheckpoint(
    'model_weights.h5', monitor='val_accuracy', save_weights_only=True, mode='max', verbose=1)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', factor=0.1, patience=2, min_lr=0.0001, mode='auto')


history = model.fit(train_generator, epochs=epochs, validation_data=validation_generator,
                    validation_steps=validation_steps, callbacks=[checkpoint, reduce_lr])

# Save the model using the model.save() function
model.save('my_disease')

# Load the saved model using the tf.keras.models.load_model() function
loaded_model = tf.keras.models.load_model('my_disease')
print(type(loaded_model))
loaded_model.evaluate(test_generator)
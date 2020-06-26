import tensorflow as tf
import pandas as pd
import pdb
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pdb
import os
import numpy as np
import matplotlib.pyplot as plt

TRAIN_DATA_PATH ="shopee-product-detection-dataset/train/train/"
batch_size = 128
epochs = 100
IMG_HEIGHT = 512
IMG_WIDTH = 512
total_train = 0

for base, dirs, files in os.walk(TRAIN_DATA_PATH):
    for Files in files:
        if (Files[0] != ".") and (len(Files) < 36):
          pdb.set_trace()
        total_train += 1

print(f"Total train {total_train}")

def preprocess_generator():
  train_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our training data
  return train_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=TRAIN_DATA_PATH,
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='binary')
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

train_data_gen = preprocess_generator()

sample_training_images, _ = next(train_data_gen)

model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
    MaxPooling2D(),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.summary()
checkpoint_path = "model/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
model.fit_generator(
    train_data_gen,
    steps_per_epoch=total_train // batch_size,
    epochs=epochs,
)
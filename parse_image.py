import numpy as np
import tensorflow as tf
import pandas as pd

# print("hello twins")

# np.random.seed(0)

# input_image = np.random.random([1, 5, 5, 1])

# conv_layer = tf.keras.layers.Conv2D(filters=3, kernel_size=(3, 3), strides=(1, 1), padding='valid')

# output = conv_layer(input_image)

# print("Shape of output:", output.shape)
# print("Output of the convolution:", output.numpy())

training_data_path = "../data/train/_annotations.csv"
training_file = pd.read_csv(training_data_path)

# insert function to grab all images but as boxes

model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer="RMSprop", loss="sparse_categorical_crossentropy", metrics=['accuracy'])
model.fit()

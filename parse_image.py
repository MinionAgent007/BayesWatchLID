import numpy as np
import tensorflow as tf
import pandas as pd
import csv
import os
from PIL import Image

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

names = []
annot = []

file_path=("./data/test/_annotations.csv")
with open(file_path, newline='') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        annot.append(row)

folder_path="./data/test"
for entry_name in os.listdir(folder_path):
    full_path = os.path.join(folder_path, entry_name)
    if(not entry_name == "_annotations.csv"):
        names.append(entry_name)

images = []
labels = []

for i in range(len(annot)):
    for j in range(len(names)):
        try:
            index = annot[i].index(names[j])

            path = "./data/test/" + annot[i][0]
            original_image = Image.open(path)
            box = (int(annot[i][4]), int(annot[i][5]), int(annot[i][6]), int(annot[i][7]))
            cropped_image = original_image.crop(box)
            images.append(cropped_image)
            labels.append(annot[i][3])
            #print(cropped_image)
        except ValueError:
            continue

print(images)
print(labels)
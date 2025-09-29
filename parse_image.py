import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import numpy as np
import tensorflow as tf
import csv
import os
from PIL import Image


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

# print(images)
# print(labels)

# resize images

target_size  = (100, 100)
processed_imgs = []

for img in images:
    img_resized = img.resize(target_size)
    img_rgb = img_resized.convert('RGB')
    processed_imgs.append(img_rgb)

images_np = np.array(processed_imgs, dtype=np.float32) / 255.0

unique_labels = sorted(list(set(labels)))
label_to_int = {label: i for i, label in enumerate(unique_labels)}
labels_as_integers = [label_to_int[label] for label in labels]

labels_np = np.array(labels_as_integers)


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
model.fit(images_np, labels_np, epochs=3)
training_loss, training_accuracy = model.evaluate(images_np, labels_np)
print ('Training loss: {}, Training accuracy: {}'.format(training_loss, training_accuracy*100))
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

print(images)
print(labels)
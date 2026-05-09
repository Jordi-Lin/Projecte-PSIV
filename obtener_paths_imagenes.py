import os
import json

PATH = os.path.join(os.getcwd(), 'Datasets', 'EMNIST', 'train')

image_paths = []

for root, dirs, files in os.walk(PATH):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            full_path = os.path.join(root, file)
            relative_path = os.path.relpath(full_path, PATH)
            image_paths.append(relative_path)

with open('EMNIST.json', 'w') as f:
    json.dump(image_paths, f, indent=4)
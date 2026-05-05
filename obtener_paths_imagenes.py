import os
import json

PATH = r"D:\Biblioteca\Desktop\PISV\Projecte\Datasets\IIIT5K\train"

image_paths = []

for root, dirs, files in os.walk(PATH):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_paths.append(file)
            
with open('paths_imagenes_faciles.json', 'w') as f:
    json.dump(image_paths, f, indent=4)
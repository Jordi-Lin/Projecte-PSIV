import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
from skimage import morphology, measure
import numpy as np
import json

PATH = os.path.join(os.getcwd(), 'Datasets', 'IIIT5K', 'train')
JSON_PATH = os.path.join(os.getcwd(), 'IIIT5K.json')

try:
    with open(JSON_PATH, 'r') as f:
        paths = json.load(f)
except FileNotFoundError:
    print(f"Error: No se encontró el archivo {JSON_PATH}")
    paths = []

total_images = 0

for root, dirs, files in os.walk(PATH):
    image_files = [
        f for f in files
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]

    total_images += len(image_files)

for j in range(total_images):

    img_path = os.path.join(PATH, paths[j])

    img = cv2.imread(img_path)

    if img is None:
        continue

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_rgb_big = cv2.resize(
        img_rgb,
        None,
        fx=3,
        fy=3,
        interpolation=cv2.INTER_CUBIC
    )

    img_gray = cv2.cvtColor(img_rgb_big, cv2.COLOR_RGB2GRAY)

    img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)

    # Binary image
    at = cv2.adaptiveThreshold(
        img_blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        21,
        4
    ) > 0

    at = morphology.remove_small_objects(at, min_size=100)

    final_image = cv2.morphologyEx(
        at.astype(np.uint8) * 255,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)),
        iterations=2
    )

    final_image2 = cv2.morphologyEx(
        final_image,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)),
        iterations=1
    )

    labels = measure.label(final_image2, connectivity=2)

    regions = measure.regionprops(labels)

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.imshow(img_rgb_big)

    for region in regions:

        # Ignore tiny regions
        if region.area < 200:
            continue

        minr, minc, maxr, maxc = region.bbox

        width = maxc - minc
        height = maxr - minr

        # Optional filtering
        if width < 5 or height < 10:
            continue

        aspect_ratio = width / height

        # Ignore weird shapes
        if aspect_ratio > 3 or aspect_ratio < 0.1:
            continue

        rect = mpatches.Rectangle(
            (minc, minr),
            width,
            height,
            fill=False,
            edgecolor='red',
            linewidth=2
        )

        ax.add_patch(rect)

    plt.title("Detected Letters")
    plt.axis('off')
    plt.show()
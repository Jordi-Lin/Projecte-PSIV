import os
import cv2
import numpy as np
from torchvision.datasets import EMNIST
from collections import defaultdict

# ---------------- CONFIGURACIÓ ----------------

OUTPUT_DIR = "emnist_custom"

TRAIN_PER_CLASS = 20
TEST_PER_CLASS = 2

IMG_SIZE = 28

# ---------------- MAPEIG DE CLASSES ----------------
# EMNIST ByClass:
# 0-9   -> digits
# 10-35 -> A-Z
# 36-61 -> a-z

def get_class_name(label):
    """
    Converteix label numèrica EMNIST -> caràcter real.
    """

    if 0 <= label <= 9:
        return str(label)

    elif 10 <= label <= 35:
        return chr(label - 10 + ord('A'))

    elif 36 <= label <= 61:
        return chr(label - 36 + ord('a'))

    return None


# ---------------- CARREGAR DATASETS ----------------

train_dataset = EMNIST(
    root="datasets/",
    split="byclass",
    train=True,
    download=True
)

test_dataset = EMNIST(
    root="datasets/",
    split="byclass",
    train=False,
    download=True
)

# ---------------- FUNCIÓ DE GUARDAT ----------------

def save_subset(dataset, split_name, max_per_class):

    counters = defaultdict(int)

    split_dir = os.path.join(OUTPUT_DIR, split_name)
    os.makedirs(split_dir, exist_ok=True)

    for i in range(len(dataset)):

        image, label = dataset[i]

        class_name = get_class_name(label)

        if class_name is None:
            continue

        # Limitar nombre d’imatges per classe
        if counters[class_name] >= max_per_class:
            continue

        # Convertir PIL -> numpy
        img = np.array(image)

        # Corregir orientació EMNIST
        img = np.rot90(img, k=3)
        img = np.fliplr(img)

        # Crear carpeta de la classe
        class_dir = os.path.join(split_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)

        # Guardar PNG
        filename = f"{counters[class_name]}.png"

        cv2.imwrite(
            os.path.join(class_dir, filename),
            img
        )

        counters[class_name] += 1

        # Comprovar si ja tenim totes les classes completes
        done = True

        for c in counters:
            if counters[c] < max_per_class:
                done = False
                break

        # 62 classes totals
        if done and len(counters) == 62:
            break

    print(f"\nSplit: {split_name}")

    total = 0

    for c in sorted(counters.keys()):
        print(f"{c}: {counters[c]}")
        total += counters[c]

    print(f"TOTAL {split_name}: {total}")


# ---------------- GENERAR TRAIN ----------------

save_subset(
    train_dataset,
    split_name="train",
    max_per_class=TRAIN_PER_CLASS
)

# ---------------- GENERAR TEST ----------------

save_subset(
    test_dataset,
    split_name="test",
    max_per_class=TEST_PER_CLASS
)

print("\nDataset generat correctament.")